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

static Scene *hst_scene;
static glm::vec3 *dev_image;
static Ray* dev_rays;
static Geom* dev_geoms;
static Material* dev_materials;
static glm::vec3* dev_colors;

static BounceRay* dev_brays;
// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above

	const Geom* geoms = &(hst_scene->geoms)[0];
	const Material* materials = &(hst_scene->materials)[0];

	const int numObjects = hst_scene->geoms.size();
	cudaMalloc((void**)&dev_rays, pixelcount*sizeof(Ray));
	cudaMalloc((void**)&dev_colors, pixelcount*sizeof(glm::vec3));
	cudaMalloc((void**)&dev_brays, pixelcount*sizeof(BounceRay));

	cudaMalloc((void**)&dev_geoms, numObjects*sizeof(Geom));
	cudaMalloc((void**)&dev_materials, numObjects*sizeof(Material));

	cudaMemcpy(dev_geoms, geoms, numObjects*sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_materials, materials, numObjects*sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    // TODO: clean up the above static variables
	cudaFree(dev_rays);
	cudaFree(dev_colors);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
    checkCUDAError("pathtraceFree");

}

__global__ void initRays(int iter, Camera cam, Ray* rays, glm::vec3* colors){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y){
		int index = x + (y * cam.resolution.x);
		glm::vec3 left = glm::cross(cam.up, cam.view);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

		float res2x = cam.resolution.x / 2.0;
		float res2y = cam.resolution.y / 2.0;

		float magx = -(res2x - x + u01(rng))*sin(cam.fov.x) / res2x;
		float magy = (res2y - y + u01(rng))*sin(cam.fov.y) / res2y;

		glm::vec3 direction = cam.view + magx*left + magy*cam.up;

		rays[index].origin = cam.position;
		rays[index].direction = direction;
		colors[index] = glm::vec3(1.0, 1.0, 1.0);

	}
}

__global__ void intersect(int iter, Camera cam, Ray* rays, glm::vec3* colors, int numObjects, const Geom* geoms, const Material* materials){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;


	if (x < cam.resolution.x && y < cam.resolution.y){
		int index = x + (y * cam.resolution.x);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

		Ray ray = rays[index];

		glm::vec3 normal;
		glm::vec3 intersectionPoint;
		float isIntersection;

		glm::vec3 minNormal;
		glm::vec3 minIntersectionPoint;
		float minDist = INFINITY;
		int obj_index = -1;
		bool outside = true;

		for (int i = 0; i < numObjects; i++){
			if (geoms[i].type == SPHERE){
				isIntersection = sphereIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
			}
			else {
				isIntersection = boxIntersectionTest(geoms[i], ray, intersectionPoint, normal, outside);
			}

			if (isIntersection > 0 && minDist > glm::distance(ray.origin, intersectionPoint)){
				minNormal = normal;
				minIntersectionPoint = intersectionPoint;
				minDist = glm::distance(ray.origin, intersectionPoint);
				obj_index = i;
			}
		}

		if (obj_index >= 0){
			//Material c = materials[index];
			//Geom g = geoms[index];
			scatterRay(rays[index], colors[index], minIntersectionPoint, minNormal, materials[geoms[obj_index].materialid], rng);
			//colors[index] = glm::vec3(0.0,1.0,0.0);
		}
		else{
			colors[index] = glm::vec3(0.0);
		}
		//image[index] = minDist == INFINITY ? glm::vec3(1.0,1.0,1.0) : glm::vec3(1.0,255.0,1.0);
	}
}

__global__ void updatePixels(Camera cam, glm::vec3* colors, glm::vec3* image){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		image[index] += colors[index];
	}
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int numObjects = hst_scene->geoms.size();
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
	//Ray* rays = (Ray*)malloc(pixelcount*sizeof(Ray));

	initRays<<<blocksPerGrid, blockSize>>>(iter, cam, dev_rays, dev_colors);
	//cudaDeviceSynchronize();

	intersect << <blocksPerGrid, blockSize >> >(iter, cam, dev_rays, dev_colors, numObjects, dev_geoms, dev_materials);

	//cudaDeviceSynchronize();

	updatePixels<<<blocksPerGrid, blockSize>>>(cam, dev_colors, dev_image);
    //generateStaticDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
