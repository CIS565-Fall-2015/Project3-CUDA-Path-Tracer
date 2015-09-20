#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/thrust.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// TODO: Why did this start throwing a multi define error?
#define checkCUDAError(msg) checkCUDAErrorFn1(msg, FILENAME, __LINE__)
void checkCUDAErrorFn1(const char *msg, const char *file, int line) {
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

__host__ __device__ thrust::default_random_engine random_engine(
	int iter, int index = 0, int depth = 0) {
	int h = utilhash((1 << 31) | (depth << 20) | iter) ^ utilhash(index);
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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Ray* dev_rays = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const Geom *geoms = &(hst_scene->geoms)[0];
	const Material *materials = &(hst_scene->materials)[0];
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
	cudaMemset(dev_rays, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_geoms, pixelcount * sizeof(Geom));
	cudaMemcpy(dev_geoms, geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, pixelcount * sizeof(Material));
	cudaMemcpy(dev_materials, materials, hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image); // no-op if dev_image is null
	cudaFree(dev_rays);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);

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

		thrust::default_random_engine rng = random_engine(iter, index, 0);
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
 * Creates a ray through each pixel on the screen.
  * Depth of Field: http://mzshehzanayub.blogspot.com/2012/10/gpu-path-tracer.html
 */
__global__ void InitializeRays(Camera cam, int iter, Ray* rays) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x); // index in ray array
	
	thrust::default_random_engine rng = random_engine(iter, index, 0);
	thrust::uniform_int_distribution<float> u01(0.0f, 1.0f);
	thrust::uniform_int_distribution<float> uHalf(-0.5f, 0.5f);

	if (cam.dof) {
		// Depth of field
		glm::vec3 horizontal, middle, vertical;
		glm::vec3 pointOnUnitImagePlane, pointOnTrueImagePlane;
		float jitteredX, jitteredY;

		// Compute point on image plane, then plane at focal distance
		horizontal = glm::cross(cam.view, cam.up) * glm::sin(cam.fov.x);
		vertical = glm::cross(glm::cross(cam.view, cam.up), cam.view) * glm::sin(-cam.fov.y);
		middle = cam.position + cam.view;

		jitteredX = (uHalf(rng) + x) / (cam.resolution.x - 1);
		jitteredY = (uHalf(rng) + y) / (cam.resolution.y - 1);
		pointOnUnitImagePlane = middle + (((2.0f * jitteredX) - 1.0f) * horizontal) + (((2.0f * jitteredY) - 1.0f) * vertical);
		pointOnTrueImagePlane = cam.position + ((pointOnUnitImagePlane - cam.position) * cam.focalDistance);

		 // Sample a random point on the lense
		float angle = TWO_PI * u01(rng);
		float distance = cam.apertureRadius * glm::sqrt(u01(rng));
		glm::vec2 aperture(glm::cos(angle) * distance, glm::sin(angle) * distance);

		rays[index].origin = cam.position + (aperture.x * glm::cross(cam.view, cam.up) + (aperture.y * glm::cross(glm::cross(cam.view, cam.up), cam.view)));;
		rays[index].direction = glm::normalize(pointOnTrueImagePlane - rays[index].origin);
	}
	else {
		//No depth of field
		glm::vec3 v = glm::cross(cam.up, cam.view);
		float halfResX, halfResY;
		float magnitudeX, magnitudeY;

		halfResX = cam.resolution.x / 2.0f;
		halfResY = cam.resolution.y / 2.0f;
		magnitudeX = (-(halfResX - x + uHalf(rng)) * sin(cam.fov.x)) / halfResX;
		magnitudeY = ((halfResY - y + uHalf(rng)) * sin(cam.fov.y)) / halfResY;

		rays[index].origin = cam.position;
		rays[index].direction = cam.view + magnitudeX * v + magnitudeY * cam.up;
	}

	rays[index].color = glm::vec3(1.0f);
	rays[index].pixel_index = index;
	rays[index].alive = true;
}

/**
* Traces an individual array for one bounce.
*/
__global__ void TraceBounce(Camera cam, int iter, int depth, glm::vec3 *image, Ray *rays, const Geom *geoms, const int numberOfObjects, const Material *materials) {
	// Thread index corresponds to the ray, pixel index is saved member of the ray
	int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int pixelIndex = rays[index].pixel_index, minGeomIndex = -1;
	float t = -1.0f, minT = FLT_MAX;
	glm::vec3 minNormal, minIntersectionPoint;
	thrust::default_random_engine rng = random_engine(iter, pixelIndex, depth);

	for (int i = 0; i < numberOfObjects; i++) {
		glm::vec3 normal, intersectionPoint;

		if (geoms[i].type == CUBE) {
			t = boxIntersectionTest(geoms[i], rays[index], intersectionPoint, normal);
		}
		else if (geoms[i].type == SPHERE) {
			t = sphereIntersectionTest(geoms[i], rays[index], intersectionPoint, normal);
		}
		else {
			// Error. will add triangles later
		}

		// Find the closest intersection
		if (t != -1.0f && minT > t) {
			minT = t;
			minNormal = normal;
			minIntersectionPoint = intersectionPoint;
			minGeomIndex = i;
		}
	}

	if (minGeomIndex == -1) {
		// Nothing was hit
		rays[index].alive = false;
		image[pixelIndex] += glm::vec3(0.0f);
	}
	else {
		int materialIndex = geoms[minGeomIndex].materialid;

		// Either we hit a light, or we scatter again
		if (materials[materialIndex].emittance > EPSILON) {
			rays[index].alive = false;
			image[pixelIndex] += rays[index].color * materials[materialIndex].color * materials[materialIndex].emittance;
		}
		else {
			scatterRay(rays[index], rays[index].color, minIntersectionPoint, minNormal, materials[materialIndex], rng);
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
	const int numberOfObjects = hst_scene->geoms.size();
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	const int blockSideLength = 8;
	const int blockSideLengthSquare = pow(blockSideLength, 2);
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
	//   * Compute one new (ray, color) pair along each path - note
	//     that many rays will terminate by hitting a light or nothing at all.
	//     You'll have to decide how to represent your path rays and how
	//     you'll mark terminated rays.
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

	InitializeRays<<<blocksPerGrid, blockSize>>>(cam, iter, dev_rays);
	checkCUDAError("InitializeRays");

	int currentDepth = 0;
	Ray *dev_raysEnd = dev_rays + pixelcount;
	while (dev_raysEnd != dev_rays && currentDepth < traceDepth) {
		int threadsRemaining = dev_raysEnd - dev_rays;
		dim3 thread_blocksPerGrid = (threadsRemaining + blockSideLengthSquare - 1) / blockSideLengthSquare;

		TraceBounce<<<thread_blocksPerGrid, blockSize>>>(cam, iter, currentDepth, dev_image, dev_rays, dev_geoms, numberOfObjects, dev_materials);
		checkCUDAError("TraceBounce");

		dev_raysEnd = StreamCompaction::Thrust::compact(dev_rays, dev_raysEnd);
		currentDepth++;
	}

	// TODO: If you add a background color that is not black, you will need to make sure to add it to any non terminated rays here.

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);
	checkCUDAError("sendImageToPBO");

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy");
}
