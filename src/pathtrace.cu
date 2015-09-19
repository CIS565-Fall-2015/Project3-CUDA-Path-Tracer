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

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    return thrust::default_random_engine(utilhash((index + 1) * iter) ^ utilhash(depth));
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
// TODO: static variables for device memory, scene/camera info, etc
// ...

static Camera *dev_camera = NULL;
static Geom *dev_geoms = NULL;
static int* dev_geoms_count = NULL;
static Material *dev_materials = NULL;
static RenderState *dev_state = NULL;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    //std::vector<Geom> geoms = hst_scene->geoms;
    //std::vector<Material> materials = hst_scene->materials;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above


    //Copy Camera
    cudaMalloc((void**)&dev_camera, sizeof(Camera));
    cudaMemcpy(dev_camera, &hst_scene->state.camera, sizeof(Camera), cudaMemcpyHostToDevice);

    //Copy geometry
    cudaMalloc((void**)&dev_geoms, hst_scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, hst_scene->geoms.data(), hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    //Copy geometry count
    int geom_count = hst_scene->geoms.size();
    cudaMalloc((void**)&dev_geoms_count, sizeof(int));
    cudaMemcpy(dev_geoms_count, &geom_count, sizeof(int), cudaMemcpyHostToDevice);

    //Copy material
    cudaMalloc((void**)&dev_materials, hst_scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, hst_scene->materials.data(), hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    //Copy state
    cudaMalloc((void**)&dev_state, sizeof(RenderState));
    cudaMemcpy(dev_state, &hst_scene->state, sizeof(RenderState), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

	cudaFree(dev_image);
    // TODO: clean up the above static variables

    cudaFree(dev_camera);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_state);
    cudaFree(dev_geoms_count);

    checkCUDAError("pathtraceFree");
}

/**
 * Example function to generate static and test the CUDA-GL interop.
 * Delete this once you're done looking at it!
 */
__global__ void generateStaticDeleteMe(Camera cam, int iter, glm::vec3 *image) {
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

//Kernel function that gets all the ray directions
__global__ void kernGetRayDirections(Camera * camera, RayState* rays, int iter)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < camera->resolution.x && y < camera->resolution.y)
	{
		int index = x + (y * camera->resolution.x);

		//TODO : Tweak the random variable here if the image looks fuzzy
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 0.005);

		//Find the ray direction
		float sx = float(x) / ((float) (camera->resolution.x) - 1.0f);
		float sy = float(y) / ((float) (camera->resolution.y) - 1.0f);

		glm::vec3 rayDir = (camera->M + (2.0f*sx - 1.0f + u01(rng)) * camera->H - (2.0f*sy - 1.0f + u01(rng)) * camera->V);
		rayDir -= camera->position;
		rayDir = glm::normalize(rayDir);

		rays[index].ray.direction = rayDir;
		rays[index].ray.origin = camera->position;
		rays[index].isAlive = true;
		rays[index].rayColor = glm::vec3(1);
		rays[index].pixelIndex = index;

//		printf("%d %d : %f %f %f\n", x, y, rayDir.x, rayDir.y, rayDir.z);
	}
}

//Kernel function that performs one iteration of tracing the path.
__global__ void kernTracePath(Camera * camera, RayState *ray, Geom * geoms, int *geomCount, Material* materials, glm::vec3* image)
{
	 int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	 int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	 if (x < camera->resolution.x && y < camera->resolution.y)
	 {
		 int index = x + (y * camera->resolution.x);

		 glm::vec3 intersectionPoint, normal;
		 float min_t = FLT_MAX, t;
		 RayState r = ray[index];
		 int nearestIndex = -1;
		 glm::vec3 nearestIntersectionPoint, nearestNormal;

		 //Find geometry intersection
		 for(int i=0; i<(*geomCount); ++i)
		 {
			 if(geoms[i].type == CUBE)
			 {
				 t = boxIntersectionTest(geoms[i], r.ray, intersectionPoint, normal);
			 }

			 else if(geoms[i].type == SPHERE)
			 {
				 t = sphereIntersectionTest(geoms[i], r.ray, intersectionPoint, normal);
			 }

			 if(t < min_t && t > 0)
			 {
				 min_t = t;
				 nearestIntersectionPoint = intersectionPoint;
				 nearestIndex = i;
				 nearestNormal = normal;
			 }
		 }

		 //If the nearest index remains unchanged, means no intersection and we can kill the ray.
		 if(nearestIndex == -1)
		 {
			 ray->isAlive = false;
			 image[r.pixelIndex] += glm::vec3(0);
		 }

		 //else find the material color
		 else
		 {

			 image[r.pixelIndex] += glm::vec3(1);
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

    const int blockSideLength = 8;
    const dim3 blockSize(blockSideLength, blockSideLength);
    const dim3 blocksPerGrid(
            (cam.resolution.x + blockSize.x - 1) / blockSize.x,
            (cam.resolution.y + blockSize.y - 1) / blockSize.y);

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    // * For each depth:
    //   * Compute one new (ray, color) pair along each path (using scatterRay).
    //     Note that many rays will terminate by hitting a light or hitting
    //     nothing at all. You'll have to decide how to represent your path rays
    //     and how you'll mark terminated rays.
    //   * Add all of the terminated rays' results into the appropriate pixels.
    //   * Stream compact away all of the terminated paths.
    //     You may use your implementation or `thrust::remove_if` or its
    //     cousins.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    // TODO: perform one iteration of path tracing

    RayState* dev_rays = NULL;

    cudaMalloc((void**)&dev_rays, pixelcount * sizeof(RayState));

    //Setup initial rays
    kernGetRayDirections<<<blocksPerGrid, blockSize>>>(dev_camera, dev_rays, iter);

    for(int i=0; i<traceDepth; ++i)
    {
    	//Take one step, should make dead rays as false
    	kernTracePath<<<blocksPerGrid, blockSize>>>(dev_camera, dev_rays, dev_geoms, dev_geoms_count, dev_materials, dev_image);

    	//Compact rays
    	thrust::remove_if(thrust::)
    }


//        generateStaticDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    cudaFree(dev_rays);

    checkCUDAError("pathtrace");
}
