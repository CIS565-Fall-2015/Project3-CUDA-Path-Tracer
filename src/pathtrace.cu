#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <glm/gtc/matrix_transform.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

struct is_terminated
{
 __host__ __device__
 bool operator() (const Ray r)
 {
	return r.terminated;
 }
};


void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_mats = NULL;
static Ray *dev_rayArray = NULL;

// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
	const Geom *geoms = &(hst_scene->geoms)[0];
	const Material *mats = &(hst_scene->materials)[0];

	cudaMalloc(&dev_geoms, pixelcount * sizeof(Geom));
	cudaMalloc(&dev_mats, pixelcount * sizeof(Material));
	cudaMalloc(&dev_rayArray, pixelcount * sizeof(Ray));
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_rayArray, 0, pixelcount * sizeof(Ray));

	cudaMemcpy(dev_geoms, geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mats, mats, hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);


    // TODO: initialize the above static variables added above

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    // TODO: clean up the above static variables
	cudaFree(dev_geoms);
	cudaFree(dev_mats);
	cudaFree(dev_rayArray);
    checkCUDAError("pathtraceFree");
}

/**
 * Example function to generate static and test the CUDA-GL interop.
 * Delete this once you're done looking at it!
 */
__global__ void generateNoiseDeleteMe(Camera cam, int iter, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // CHECKITOUT: Note that on every iteration, noise gets added onto
        // the image (not replaced). As a result, the image smooths out over
        // time, since the output image is the contents of this array divided
        // by the number of iterations.
        //
        // Your renderer will do the same thing, and, over time, it will become
        // smoother.
        image[index] += glm::vec3(u01(rng));
    }
}

//Create ray to be shot at a pixel in the image
__global__ void kernRayGenerate(Camera cam, Ray *ray, int iter, bool dof){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y*cam.resolution.x);
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> unitDistrib(-.5f, .5f);
	thrust::uniform_real_distribution<float> dofDistrib(-1.0f, 1.0f);
	//Calculate camera's world position
	if (x < cam.resolution.x && y < cam.resolution.y) {
		glm::vec3 A = glm::cross(cam.view, cam.up);
		glm::vec3 B = glm::cross(A, cam.view);
		glm::vec3 M = cam.position + cam.view;
		float lenC = glm::length(cam.view);
		float lenA = glm::length(A);
		float lenB = glm::length(B);
		float tantheta = (float)cam.resolution.x;
		tantheta /= (float)cam.resolution.y;
		tantheta *= tan((float)glm::radians(cam.fov[1]));
	
		glm::vec3 H = (A*lenC*tantheta) / lenA;
		glm::vec3 V = (B*lenC*tan((float)glm::radians(cam.fov[1]))) / lenB;

		//Create ray with direction and origin
		//Jitter rays with uniform distribution
		//printf("%f ", unitDistrib(rng));
		float sx = ((float)x + unitDistrib(rng)) / ((float)cam.resolution.x - 1.0f);
		float sy = ((float)y + unitDistrib(rng)) / ((float)cam.resolution.y - 1.0f);
		//Get world coordinates of pixel
		glm::vec3 WC = M - (2.0f*sx - 1.0f)*H - (2.0f*sy - 1.0f)*V;
		//Get direction of ray
		glm::vec3 dir = glm::normalize(WC - cam.position);

		ray[index].origin = cam.position;
		ray[index].direction = dir;
		ray[index].color = glm::vec3(1.0, 1.0, 1.0);
		ray[index].index = index;
		ray[index].terminated = false;

		if (dof == true) {
			glm::vec3 apOff = glm::vec3(dofDistrib(rng), dofDistrib(rng), 0.0f);
			glm::vec3 new_E = cam.position + apOff;
			float focal = 11.587f; //glm::length(glm::vec3(-2.0f, 5.0f,2.0f) - new_E);
			dir *= focal;
			dir -= apOff;
			dir = glm::normalize(dir);
			ray[index].origin = new_E;
			ray[index].direction = dir;
		}
	}
}

//Helper function to get random point on cubic light
__device__ glm::vec3 getRandomPointOnCube(Geom node, int iter, int index) {
	// TODO: get the dimensions of the transformed cube in world space
	glm::vec3 dim(0.0f, 0.0f, 0.0f);
	dim = node.scale;

	// Get surface area of the cube
	float side1 = dim[0] * dim[1];		// x-y
	float side2 = dim[1] * dim[2];		// y-z
	float side3 = dim[0] * dim[2];		// x-z
	float totalArea = 2.0f * (side1 + side2 + side3);	

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> unitDistrib(-.5f, .5f);
	thrust::uniform_real_distribution<float> dofDistrib(0.0f, 1.0f);

	// pick random face weighted by surface area
	float r = floor(dofDistrib(rng));
	// pick 2 random components for the point in the range (-0.5, 0.5)
	float c1 = unitDistrib(rng);
	float c2 = unitDistrib(rng);

	glm::vec3 point;
	if (r < side1 / totalArea) {				
		// x-y front
		point = glm::vec3(c1, c2, 0.5f);
	} else if (r < (side1 * 2) / totalArea) {
		// x-y back
		point = glm::vec3(c1, c2, -0.5f);
	} else if (r < (side1 * 2 + side2) / totalArea) {
		// y-z front
		point = glm::vec3(0.5f, c1, c2);
	} else if (r < (side1 * 2 + side2 * 2) / totalArea) {
		// y-z back
		point = glm::vec3(-0.5f, c1, c2);
	} else if (r < (side1 * 2 + side2 * 2 + side3) / totalArea) {
		// x-z front 
		point = glm::vec3(c1, 0.5f, c2);
	} else {
		// x-z back
		point = glm::vec3(c1, -0.5f, c2);
	}

	// TODO: transform point to world space
	glm::mat4 T(1.0f);
	T = glm::translate(T, node.translation);
				
	if (node.rotation[0] != 0){
		T = glm::rotate(T, node.rotation[0]*(PI/180.0f), glm::vec3(1,0,0));
	}
	if (node.rotation[1] != 0){
		T = glm::rotate(T, node.rotation[1]*(PI/180.0f), glm::vec3(0,1,0));
	}
	if (node.rotation[2] != 0){
		T = glm::rotate(T, node.rotation[2]*(PI/180.0f), glm::vec3(0,0,1));
	}
				
	//T = glm::scale(T, node.scale);
	glm::vec4 newPoint = T*glm::vec4(point, 1.0f);
	point = glm::vec3(newPoint[0], newPoint[1], newPoint[2]);
	return point;
}

//Helper function to get random point on spherical light
/*__device__ glm::vec3 getRandomPointOnSphere(Geom node, int iter, int index) {
	// generate u, v, in the range (0, 1)
	float u = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float v = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

	float theta = 2.0f * PI * u;
	float phi = acos(2.0f * v - 1.0f);

	// find x, y, z coordinates assuming unit sphere in object space
	glm::vec3 point;
	point[0] = sin(phi) * cos(theta);
	point[1] = sin(phi) * sin(theta);
	point[2] = cos(phi);

	// TODO: transform point to world space
	glm::mat4 T(1.0f);
	T = glm::translate(T, node.translation);
				
	if (node.rotation[0] != 0){
		T = glm::rotate(T, node.rotation[0]*(PI/180.0f), glm::vec3(1,0,0));
	}
	if (node.rotation[1] != 0){
		T = glm::rotate(T, node.rotation[1]*(PI/180.0f), glm::vec3(0,1,0));
	}
	if (node.rotation[2] != 0){
		T = glm::rotate(T, node.rotation[2]*(PI/180.0f), glm::vec3(0,0,1));
	}
				
	glm::vec4 newPoint = T*glm::vec4(point, 1.0f);
	point = glm::vec3(newPoint[0], newPoint[1], newPoint[2]);
	return point;
}*/
//Helper function to find closest intersection
__device__ float closestIntersection(Ray ray, const Geom* geoms, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, int &objIndex, const int numGeoms){
	glm::vec3 interPoint;
	glm::vec3 norm;
	bool out;
	float t = -1;
	float dist;
	for (int i = 0; i < numGeoms; i++) {	
		if (geoms[i].type == CUBE) {
			dist = boxIntersectionTest(geoms[i], ray, interPoint, norm, out);
		}
		else if (geoms[i].type == SPHERE) {
			dist = sphereIntersectionTest(geoms[i], ray, interPoint, norm, out);
		}
		if ((dist != -1 && dist < t) || t == -1) {
			t = dist;
			intersectionPoint = interPoint;
			normal = norm;
			outside = out;
			objIndex = i;
		
		}
	}
	return t;
		
}

//Function to find next ray
__global__ void kernPathTracer(Camera cam, Ray* rayArray, const Geom* geoms, const Material* mats, const int numGeoms, const int numMats, glm::vec3* dev_image, int iter, int depth, int traceDepth){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	//find closest intersection
	if (x < cam.resolution.x && y < cam.resolution.y && rayArray[index].terminated == false) {
		
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
		glm::vec3 interPoint;
		glm::vec3 norm;
		bool out;
		int objIndex;
		if (depth == traceDepth) {
			dev_image[rayArray[index].index] == glm::vec3(0.0f, 0.0f, 0.0f);
			for (int i = 0; i < numGeoms; i++) {
				if (mats[geoms[i].materialid].emittance > 0 && mats[geoms[rayArray[index].geomid].materialid].emittance == 0 && mats[geoms[rayArray[index].geomid].materialid].hasReflective == 0 && mats[geoms[rayArray[index].geomid].materialid].hasRefractive == 0) {
					glm::vec3 new_pt = getRandomPointOnCube(geoms[i], iter, index);
					rayArray[index].direction = rayArray[index].origin + glm::normalize(new_pt - rayArray[index].origin);
					float t = closestIntersection(rayArray[index], geoms, interPoint, norm, out, objIndex, numGeoms);
					if (objIndex == i) {
						printf("hit light in direct");
						rayArray[index].color *= mats[geoms[i].materialid].emittance*mats[geoms[objIndex].materialid].color;
						dev_image[index] += rayArray[index].color;
					}
				}
			}
			
		}
		float t = closestIntersection(rayArray[index], geoms, interPoint, norm, out, objIndex, numGeoms);
		rayArray[index].geomid = objIndex;
		//get direction of next ray and compute new color
		if (t >= 0.0f) {
			if (mats[geoms[objIndex].materialid].emittance >= 1) {
				rayArray[index].color *= mats[geoms[objIndex].materialid].emittance*mats[geoms[objIndex].materialid].color;
				dev_image[rayArray[index].index] += rayArray[index].color;
				rayArray[index].terminated = true;
			}
			else {
				scatterRay(rayArray[index], rayArray[index].color, interPoint, norm, mats[geoms[objIndex].materialid], rng);
			}
		}
		else {
			//dev_image[index] *= glm::vec3(0.0f, 0.0f, 0.0f); //rayArray[index].color; 
			rayArray[index].terminated = true;
		}
	}	
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
	
	int numGeoms = hst_scene->geoms.size();
	int numMats = hst_scene->materials.size();
	Ray *rayArray = new Ray[pixelcount];

    const int blockSideLength = 8;
    const dim3 blockSize(blockSideLength, blockSideLength);
    const dim3 blocksPerGrid(
            (cam.resolution.x + blockSize.x - 1) / blockSize.x,
            (cam.resolution.y + blockSize.y - 1) / blockSize.y);

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray is a (ray, color) pair, where color starts as the
    //     multiplicative identity, white = (1, 1, 1).
    //   * For debugging, you can output your ray directions as colors.
    // * For each depth:
    //   * Compute one new (ray, color) pair along each path (using scatterRay).
    //     Note that many rays will terminate by hitting a light or hitting
    //     nothing at all. You'll have to decide how to represent your path rays
    //     and how you'll mark terminated rays.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //       surface.
    //     * You can debug your ray-scene intersections by displaying various
    //       values as colors, e.g., the first surface normal, the first bounced
    //       ray direction, the first unlit material color, etc.
    //   * Add all of the terminated rays' results into the appropriate pixels.
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    
    // TODO: perform one iteration of path tracing
	bool dof = false;
	kernRayGenerate<<<blocksPerGrid, blockSize>>>(cam, dev_rayArray, iter, dof);

	for (int i = 0; i < traceDepth + 1; i++) {
		kernPathTracer<<<blocksPerGrid, blockSize>>>(cam, dev_rayArray, dev_geoms, dev_mats, numGeoms, numMats, dev_image, iter, i, traceDepth);

		//thrust::remove_if(thrust::host, dev_rayArray, dev_rayArray + pixelcount, is_terminated());
	}
	
	cudaMemcpy(rayArray, dev_rayArray, pixelcount*sizeof(Ray), cudaMemcpyDeviceToHost);
	
    //generateNoiseDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);
	
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
