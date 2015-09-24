#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

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
__global__ void kernRayGenerate(Camera cam, Ray *ray){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
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

		float sx = (float)x / ((float)cam.resolution.x - 1.0f);
		float sy = (float)y / ((float)cam.resolution.y - 1.0f);
		//Get world coordinates of pixel
		glm::vec3 WC = M - (2.0f*sx - 1.0f)*H - (2.0f*sy - 1.0f)*V;
		//Get direction of ray
		glm::vec3 dir = glm::normalize(WC - cam.position);

		ray[x + (y*cam.resolution.x)].origin = cam.position;
		ray[x + (y*cam.resolution.x)].direction = dir;
		ray[x + (y*cam.resolution.x)].color = glm::vec3(1.0, 1.0, 1.0);
		ray[x + (y*cam.resolution.x)].index = -1*(x + (y*cam.resolution.x));
	}
}

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
__global__ void kernPathTracer(Camera cam, Ray* rayArray, const Geom* geoms, const Material* mats, const int numGeoms, const int numMats, glm::vec3* dev_image, int iter){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	//find closest intersection
	if (x < cam.resolution.x && y < cam.resolution.y && rayArray[index].index < 0) {
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		glm::vec3 interPoint;
		glm::vec3 norm;
		bool out;
		int objIndex;
		float t = closestIntersection(rayArray[index], geoms, interPoint, norm, out, objIndex, numGeoms);

		//get direction of next ray and compute new color
		if (t >= 0.0f) {
			if (mats[geoms[objIndex].materialid].emittance >= 1) {
				rayArray[index].color *= mats[geoms[objIndex].materialid].emittance*mats[geoms[objIndex].materialid].color;
				dev_image[index] += rayArray[index].color;
				rayArray[index].index *= -1;
			}
			else {
				scatterRay(rayArray[index], rayArray[index].color, interPoint, norm, mats[geoms[objIndex].materialid], rng);
			}
		}
		else {
			//dev_image[index] *= glm::vec3(0.0f, 0.0f, 0.0f); //rayArray[index].color; 
			rayArray[index].index *= -1;
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
	kernRayGenerate<<<blocksPerGrid, blockSize>>>(cam, dev_rayArray);

	for (int i = 0; i < traceDepth; i++) {
		kernPathTracer<<<blocksPerGrid, blockSize>>>(cam, dev_rayArray, dev_geoms, dev_mats, numGeoms, numMats, dev_image, iter);
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
