#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <stream_compaction/efficient.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ANTIALIASING 0
#define DOF 0
#define USETHRUSTCOMPACTION 0
#define TIMEFUNCTION 0
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
#define ERRORCHECK 0
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	+ cudaDeviceSynchronize();
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
#endif ERRORCHECK
}

__host__ __device__ thrust::default_random_engine random_engine(
        int iter, int index = 0, int depth = 0) {
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
__constant__ static glm::vec3 *dev_image = NULL;
__constant__ static Geom* dev_geoms = NULL;
__constant__ static Material* dev_materials = NULL;
__constant__ static glm::vec3 *dev_oversample_image = NULL;
static int geomcount = 0;
static int oversampling_pass = 3;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_oversample_image, pixelcount * sizeof(glm::vec3));

	Geom* hst_geoms = hst_scene->geoms.data();
	Material* hst_materials = hst_scene->materials.data();

	geomcount = hst_scene->geoms.size();

	cudaMalloc((void**)&dev_geoms, hst_scene->geoms.size()*sizeof(Geom));
	cudaMalloc((void**)&dev_materials, hst_scene->materials.size()*sizeof(Material));
	cudaMemcpy(dev_geoms, hst_geoms, hst_scene->geoms.size()*sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_materials, hst_materials, hst_scene->materials.size()*sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_oversample_image);

    checkCUDAError("pathtraceFree");
}

__global__ void initRayGrid(PathRay *oGrid, const Camera cam){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		//PathRay pr;
		oGrid[index].index = index;
		oGrid[index].color = glm::vec3(1.0f);

		oGrid[index].ray.origin = cam.position;
		oGrid[index].terminate = false;
		oGrid[index].matId = -1;

		// Grid center to pixel
		float pX = x - cam.resolution.x / 2;
		float pY = cam.resolution.y / 2 - y;

		// Vector: grid center to pixel
		glm::vec3 o2px = cam.right*pX + cam.up*pY;
		//glm::vec3 o2px = glm::vec3(cam.right.x*pX + cam.up.x*pY, cam.right.y*pX + cam.up.y*pY, cam.right.z*pX + cam.up.z*pY);
		// Ray vector
		oGrid[index].ray.direction = cam.toGrid + o2px;

		//oGrid[index] = pr;

		// Ray direction debug
		//float l = glm::length(ray.ray.direction);
		//image[index] += glm::vec3(abs(ray.ray.direction.x / l), abs(ray.ray.direction.y / l), 0);
	}
}


__global__ void interesect(PathRay *grid, const Geom *iGeoms, const Camera cam, const int grid_size, const int geomcount){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	extern __shared__ Geom geoms[];

	int bIndex = threadIdx.x + threadIdx.y * blockDim.x;
	if (bIndex < geomcount){
		geoms[bIndex] = iGeoms[bIndex];
	}

	__syncthreads();

	if (index < grid_size) {
		// Intersection test
		//PathRay pr = grid[index];
		grid[index].hasIntersect = false;
		glm::vec3 iPoint(0.0f);
		glm::vec3 iNormal(0.0f);

		float rayLength = 0.0f;
		float oldLength = -1.0f;
		int idx = 0;
		bool outside = false;
		for (int i = 0; i < geomcount; ++i){
			if (geoms[i].type == SPHERE){
				rayLength = sphereIntersectionTest(geoms[i], grid[index].ray, iPoint, iNormal, outside);
			}
			else {
				rayLength = boxIntersectionTest(geoms[i], grid[index].ray, iPoint, iNormal, outside);
			}
			// Find the nearest intersection
			if (rayLength != -1.0f){
				grid[index].hasIntersect = true;
				if (oldLength == -1.0f || rayLength < oldLength){
					grid[index].intersect = iPoint;
					grid[index].normal = iNormal;
					grid[index].outside = outside;
					oldLength = rayLength;
					idx = i;
				}
			}
			/*
			Geom g = iGeoms[i];
			if (g.type == SPHERE){
				rayLength = sphereIntersectionTest(g, pr.ray, iPoint, iNormal, outside);
			} else {
				rayLength = boxIntersectionTest(g, pr.ray, iPoint, iNormal, outside);
			}
			// Find the nearest intersection
			if (rayLength != -1.0f){
				pr.hasIntersect = true;
				if (oldLength == -1.0f || rayLength < oldLength){
					oldLength = rayLength;
					pr.intersect = iPoint;
					pr.normal = iNormal;
					pr.matId = g.materialid;
					pr.outside = outside;
				}
			}
			*/
		}
		grid[index].matId = geoms[idx].materialid;
		//grid[index] = pr;
	}
};

__global__ void scatter(PathRay *grid, const Material *iMaterials, const Camera cam, const int grid_size, const int iter, const int depth){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	__syncthreads();

	if (index < grid_size) {
		//PathRay pr = grid[index];
		//Material m = iMaterials[pr.matId];
		scatterRay(grid[index], iMaterials[grid[index].matId], random_engine(iter, index, depth));
		//grid[index] = pr;
	}
};

__global__ void terminatePath(PathRay *grid, const Material *iMaterials, const Camera cam, const int grid_size){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	if (index < grid_size) {
		if (grid[index].hasIntersect){
			// Hits a light
			if (iMaterials[grid[index].matId].emittance > 0.0f){
				grid[index].terminate = true;
				grid[index].color = grid[index].color * iMaterials[grid[index].matId].color * iMaterials[grid[index].matId].emittance;
			}
		}
		else {
			// No intersections
			grid[index].terminate = true;
			grid[index].color = glm::vec3(0.0f);
		}
		//grid[index] = pr;
	}
}

__global__ void fillPixel(const PathRay *grid, glm::vec3 *image, const Camera cam, const int grid_size){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	if (index < grid_size) {
		PathRay pr = grid[index];
		if (pr.terminate){
			image[pr.index] += pr.color;
		}
	}
}

#if ANTIALIASING
__global__ void avgOversample(glm::vec3 *oImage, const glm::vec3 *tempImage, const Camera cam, const int passes){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		oImage[index] += tempImage[index] / (float)passes;
	}
}

__global__ void jitterRay(PathRay *grid, thrust::default_random_engine rng, const Camera cam){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathRay pr = grid[index];
#if DOF
		thrust::uniform_real_distribution<float> u01(-0.1, 0.1);
		// Find intersection to focal plane
		glm::vec3 p = pr.ray.origin + glm::normalize(pr.ray.direction) * cam.dof;
		glm::vec3 jitter = cam.up*u01(rng) + cam.right*u01(rng);
		// Jitter ray origin
		pr.ray.origin = pr.ray.origin + jitter;
		// Update ray direction based on new origin
		// So new ray always points to exact pixel cell on focal plane
		pr.ray.direction = p - pr.ray.origin;
#else
		thrust::uniform_real_distribution<float> u01(-0.01, 0.01);
		pr.ray.origin = glm::vec3(pr.ray.origin.x + u01(rng), pr.ray.origin.y + u01(rng), pr.ray.origin.z + u01(rng));
#endif DOF
		grid[index] = pr;
	}
}
#endif ANTIALIASING

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
    dim3 blocksPerGrid(
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

    // Perform one iteration of path tracing

	thrust::device_vector<PathRay> dev_grid(pixelcount);
	PathRay *dev_grid_ptr = thrust::raw_pointer_cast(&dev_grid[0]);

#if ANTIALIASING
	cudaMemset(dev_oversample_image, 0, pixelcount * sizeof(glm::vec3));
	for (int a = 0; a < oversampling_pass; a++){
		dev_grid.resize(pixelcount);
		// initRayGrid
		initRayGrid << <blocksPerGrid, blockSize >> >(dev_grid_ptr, cam);
		int grid_size = dev_grid.size();
		// Jitter for antialiasing oversampling; also accounts for DOF effect if enabled
		jitterRay << <blocksPerGrid, blockSize >> >(dev_grid_ptr, random_engine(iter, 0, oversampling_pass), cam);
#else
	// initRayGrid
	initRayGrid << <blocksPerGrid, blockSize >> >(dev_grid_ptr, cam);
	int grid_size = dev_grid.size();
#endif ANTIALIASING
	// For each traceDepth
	for (int d = 0; d < traceDepth; d++){

#if TIMEFUNCTION
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
#endif TIMEFUNCTION

		// Intersection test
		interesect << <blocksPerGrid, blockSize, geomcount*sizeof(Geom) >> >(dev_grid_ptr, dev_geoms, cam, grid_size, geomcount);
		checkCUDAError("intersect");

		// Mark all terminated paths
		terminatePath << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_materials, cam, grid_size);
		checkCUDAError("terminatePath");

		// Paint image
#if ANTIALIASING
		fillPixel << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_oversample_image, cam, grid_size);
#else
		fillPixel << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_image, cam, grid_size);
#endif ANTIALIASING
		checkCUDAError("fillPixel");

		// Stream compaction
#if USETHRUSTCOMPACTION
		thrust::detail::normal_iterator<thrust::device_ptr<PathRay>> newGridEnd = thrust::remove_if(dev_grid.begin(), dev_grid.end(), is_terminated());
		checkCUDAError("thrustCompact");
		dev_grid.erase(newGridEnd, dev_grid.end());
		grid_size = dev_grid.size(); 
#else
		grid_size = StreamCompaction::Efficient::compact(grid_size, dev_grid_ptr);
		checkCUDAError("efficientCompact");
#endif USETHRUSTCOMPACTION

		// Scatter
		scatter << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_materials, cam, grid_size, iter, d);
		checkCUDAError("scatter");

		// Stream compaction stat
		//printf("Iter: %d / Depth: %d / Grid size: %d\n", iter, d, grid_size);

#if TIMEFUNCTION
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms1 = 0;
		cudaEventElapsedTime(&ms1, start, stop);

		// Exec time comparison: thrust vs. work-efficient
		printf("Iter: %d / Depth: %d / Grid size: %d / Thrust?: %d / Time: %f\n", iter, d, grid_size, USETHRUSTCOMPACTION, ms1);
#endif TIMEFUNCTION
	}
	dev_grid.clear();
#if ANTIALIASING
	}
	// Average oversampled colors and fill into image
	avgOversample << <blocksPerGrid, blockSize >> >(dev_image, dev_oversample_image, cam, oversampling_pass);
#endif ANTIALIASING

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
