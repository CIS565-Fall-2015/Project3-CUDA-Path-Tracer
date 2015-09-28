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
static glm::vec3 *dev_image = NULL;
// TODO: static variables for device memory, scene/camera info, etc
// ...

static int hst_geomCount; // number of geometries to check against
static Geom *dev_geoms; // pointer to geometries in global memory
static Material *dev_mats; // pointer to materials in global memory
static PathRay *dev_firstBounce; // cache of the first raycast of any iteration
static PathRay *dev_rayPool; // pool of rays "in flight"
static int pixelcount;
static glm::vec3 *dev_sample = NULL;

static int *dev_compact_tmp_array; // temporary array used by compact
static int *dev_compact_scan_array; // scan array used by compact

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above
	// set up the geometries
	hst_geomCount = scene->geoms.size();
	cudaMalloc(&dev_geoms, hst_geomCount * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), sizeof(Geom) * hst_geomCount,
		cudaMemcpyHostToDevice);

	// set up the materials
	int hst_matCount = scene->materials.size();
	cudaMalloc(&dev_mats, hst_matCount * sizeof(Material));
	cudaMemcpy(dev_mats, scene->materials.data(),
		sizeof(Material) * hst_matCount, cudaMemcpyHostToDevice);

	// set up space for the first cast
	// we'll be casting a ray from every pixel
	int numPixels = scene->state.camera.resolution.x;
	numPixels *= scene->state.camera.resolution.y;
	cudaMalloc(&dev_firstBounce, numPixels * sizeof(PathRay));
	cudaMalloc(&dev_rayPool, numPixels * sizeof(PathRay));

	// allocate space for a "sample"
	cudaMalloc(&dev_sample, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_sample, 0, pixelcount * sizeof(glm::vec3));

	// allocate space for the compact temp array
	int logn = ilog2ceil(pixelcount);
	int pow2 = (int)pow(2, logn);

	// TODO: something better than just allocating up to the next power of two. inefficient.
	cudaMalloc(&dev_compact_tmp_array, pow2 * sizeof(int));
	cudaMalloc(&dev_compact_scan_array, pow2 * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    // TODO: clean up the above static variables
	cudaFree(dev_geoms);
	cudaFree(dev_mats);
	cudaFree(dev_firstBounce);
	cudaFree(dev_rayPool);
	cudaFree(dev_sample);

	cudaFree(dev_compact_tmp_array);
	cudaFree(dev_compact_scan_array);

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

__global__ void singleBounce(int iter, int pixelCount, Material* dev_mats,
	int geomCount, Geom* dev_geoms, PathRay* dev_rayPool) {
	// what information do we need for a single ray given index by block/grid thing?
	// we need:
	// - pointer to the sample that will have color updated OR color storage
	// - access to the "current rays" ray pool
	//		-is it better to read and write to the same place?
	//		-or have two buffers and flip them?
	//			-"next rays" memory should have same allocated size as current rays
	//			-should start off allocated full of "terminated" rays
	//			-terminated rays have depth of over MAX_DEPTH
	// - count of ray depth
	// - so I've added a PathRay struct that contains color and trace depth

	// 1) grab the index of the ray
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < pixelCount) {
		// 2) check what it intersects with
		Geom *nearestGeom = NULL;
		glm::vec3 isx_point;
		glm::vec3 isx_norm;
		float t = INFINITY;
		for (int i = 0; i < geomCount; i++) {
			float candidate_t = -1.0f;
			glm::vec3 candidate_isx_point;
			glm::vec3 candidate_isx_norm;
			if (dev_geoms[i].type == CUBE) {
				candidate_t = boxIntersectionTest(dev_geoms[i], dev_rayPool[index].ray,
					candidate_isx_point, candidate_isx_norm);
			}
			else if (dev_geoms[i].type == SPHERE) {
				candidate_t = sphereIntersectionTest(dev_geoms[i], dev_rayPool[index].ray,
					candidate_isx_point, candidate_isx_norm);
			}
			if (candidate_t > 0.0f && candidate_t < t) {
				t = candidate_t;
				isx_point = candidate_isx_point;
				isx_norm = candidate_isx_norm;
				nearestGeom = &dev_geoms[i];
			}
		}

		// 3) update the ray in its slot
		if (nearestGeom) {
			thrust::default_random_engine rng = random_engine(iter, index, 0);
			scatterRay(dev_rayPool[index], isx_point, isx_norm,
				dev_mats[nearestGeom->materialid], rng);
			dev_rayPool[index].depth--;
		}
		else {
			// ray cast out into space
			dev_rayPool[index].depth = 0;
			dev_rayPool[index].color = glm::vec3(0, 0, 0);
		}

		// debug: intersection check.
		// image[index] += isx_norm;

		// debug: flat color
		//if (nearestGeom) {
		//	image[index] += dev_mats[nearestGeom->materialid].color;
		//}
	}
}

// generates the initial raycasts
__global__ void rayCast(Camera cam, PathRay* dev_rayPool, int trace_depth) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// compute x and y screen coordinates by reversing int index = x + (y * resolution.x);
	int y = index / (int)cam.resolution.x;
	int x = index - y * cam.resolution.x;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);

		// generate a PathRay to cast
		glm::vec3 ref = cam.position + cam.view;
		glm::vec3 R = glm::cross(cam.up, cam.view);
		glm::vec3 V = cam.up * glm::tan(cam.fov.y * 0.01745329251f);
		glm::vec3 H = R * glm::tan(cam.fov.x * 0.01745329251f);
		// sx = ((2.0f * x) / cam.resolution.x) - 1.0f
		// sy = 1.0f - ((2.0f * y) / cam.resolution.y)
		glm::vec3 p = H * (((2.0f * x) / cam.resolution.x) - 1.0f) +
			V * (1.0f - ((2.0f * y) / cam.resolution.y)) + ref;
		dev_rayPool[index].ray.direction = glm::normalize(p - cam.position);
		dev_rayPool[index].ray.origin = cam.position;
		dev_rayPool[index].color = glm::vec3(1.0f);
		dev_rayPool[index].depth = trace_depth;
		dev_rayPool[index].pixelIndex = index;

		//glm::vec3 debug = glm::normalize(p - cam.position);
		//debug.x = abs(debug.x);
		//debug.y = abs(debug.y);
		//debug.z = abs(debug.z);
		//
		//image[index] += debug;
	}
}

// transfers colors from thread pool to the image
__global__ void poolToImage(PathRay* dev_rayPool, glm::vec3 *sample) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (dev_rayPool[index].depth <= 0) {
		sample[dev_rayPool[index].pixelIndex] = dev_rayPool[index].color;
	}
}

// transfers colors from thread pool to the image
__global__ void mergeSample(glm::vec3 *sample, glm::vec3 *image) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	image[index] += sample[index];
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	// on iteration 1, run tests for efficient shared memory scan
	if (iter == 1) {
		StreamCompaction::Efficient::scan_components_test();
	}



	// wipe the sample buffer
	cudaMemset(dev_sample, 0, pixelcount * sizeof(glm::vec3));

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

    //generateNoiseDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);

	// while there are still unfinished rays in the scene, trace.
	int unfinishedRays = cam.resolution.x * cam.resolution.y;
	dim3 iterBlockSize(blockSideLength * blockSideLength);
	dim3 iterBlocksPerGrid((unfinishedRays + iterBlockSize.x - 1) /
		iterBlockSize.x);

	rayCast <<<iterBlocksPerGrid, iterBlockSize >>>(cam, dev_rayPool, traceDepth);
	
	while (unfinishedRays > 0) {
		iterBlocksPerGrid.x = (unfinishedRays + iterBlockSize.x - 1) /
			iterBlockSize.x;

		singleBounce <<<iterBlocksPerGrid, iterBlockSize >>>(iter, pixelcount,
			dev_mats, hst_geomCount, dev_geoms, dev_rayPool);

		poolToImage << <iterBlocksPerGrid, iterBlockSize >> >(dev_rayPool, dev_sample);
		
		//unfinishedRays = cullRaysThrust(unfinishedRays);
		//unfinishedRays = cullRaysEfficient(unfinishedRays);
		unfinishedRays = cullRaysEfficientSharedMemory(unfinishedRays);
		if (iter == 1) printf("unfinished rays: %i\n", unfinishedRays);
	}

	// transfer results over to the image
	iterBlocksPerGrid.x = (cam.resolution.x * cam.resolution.y + iterBlockSize.x - 1) /
		iterBlockSize.x;

	mergeSample << <iterBlocksPerGrid, iterBlockSize >> >(dev_sample, dev_image);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

struct bottomed_out
{
	__host__ __device__
	bool operator()(const PathRay pathray)
	{
		return pathray.depth <= 0;
	}
};

// culls rays using stream compaction. for now, just uses thrust.
int cullRaysThrust(int numRays) {
	PathRay *newEnd = thrust::remove_if(thrust::device, dev_rayPool, dev_rayPool + numRays, bottomed_out());
	// get the index of newEnd
	int newNumRays = 0;
	for (int i = 0; i < numRays; i++) {
		if (&dev_rayPool[i] == newEnd) {
			newNumRays = i;
			break;
		}
	}
	return newNumRays;
}

// in parallel, compute the temp array
__global__ void tempArray(PathRay* dev_rayPool, int *dev_tmp, int numRays) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numRays) {
		dev_tmp[index] = dev_rayPool[index].depth > 0;
	}
}

// perform in-place scatter on the ray pool
__global__ void scatterRays(PathRay* dev_rayPool, int *dev_tmp, int *dev_scan, int numRays) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numRays && dev_tmp[index]) {
		dev_rayPool[dev_scan[index]] = dev_rayPool[index];
	}
}

// culls rays using work efficient global memory stream compaction
int cullRaysEfficient(int numRays) {

	// zero pad up to a power of 2
	int logn = ilog2ceil(numRays);
	int pow2 = (int)pow(2, logn);

	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength * blockSideLength, 1);
	const dim3 blocksPerGrid((pow2 + blockSize.x - 1) / blockSize.x);

	// zero out the temp array and the scan array
	cudaMemset(dev_compact_tmp_array, 0, pow2 * sizeof(int));
	cudaMemset(dev_compact_scan_array, 0, pow2 * sizeof(int));
	// Step 1: compute temporary array containing 1 if criteria met, 0 otherwise
	tempArray <<< blocksPerGrid, blockSize >>>(dev_rayPool, dev_compact_tmp_array, numRays);
	// make a copy of the temp array so we can do an in-place upsweep downsweep step. TODO: can we get around this memcpy?
	cudaMemcpy(dev_compact_scan_array, dev_compact_tmp_array, sizeof(int) * pow2, cudaMemcpyDeviceToDevice);

	// Step 2: run exclusive scan on temporary array
	StreamCompaction::Efficient::up_sweep_down_sweep(pow2, dev_compact_scan_array, blocksPerGrid.x, blockSize.x);

	// Step 3: scatter in place
	scatterRays << <blocksPerGrid, blockSize >> >(dev_rayPool, dev_compact_tmp_array, dev_compact_scan_array, numRays);

	int last_index;
	cudaMemcpy(&last_index, dev_compact_scan_array + (numRays - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	int last_true_false;
	cudaMemcpy(&last_true_false, dev_compact_tmp_array + (numRays - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	return last_index + last_true_false;
}

// perform in-place scatter on the ray pool
__global__ void inclusiveScatterRays(PathRay* dev_rayPool, int *dev_tmp, int *dev_scan, int numRays) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numRays && dev_tmp[index]) {
		if (index - 1 >= 0) {
			dev_rayPool[dev_scan[index - 1]] = dev_rayPool[index];
		}
		else {
			dev_rayPool[0] = dev_rayPool[index];
		}
	}
}

// culls rays using work efficient shared memory stream compaction
int cullRaysEfficientSharedMemory(int numRays) {
	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength * blockSideLength, 1);
	const dim3 blocksPerGrid((numRays + blockSize.x - 1) / blockSize.x);

	// zero out the temp array and the scan array
	cudaMemset(dev_compact_tmp_array, 0, numRays * sizeof(int));
	cudaMemset(dev_compact_scan_array, 0, numRays * sizeof(int));
	// Step 1: compute temporary array containing 1 if criteria met, 0 otherwise
	tempArray << < blocksPerGrid, blockSize >> >(dev_rayPool, dev_compact_tmp_array, numRays);
	// make a copy of the temp array so we can do an scatter. TODO: can we get around this memcpy?
	cudaMemcpy(dev_compact_scan_array, dev_compact_tmp_array, sizeof(int) * numRays, cudaMemcpyDeviceToDevice);

	// Step 2: run inclusive scan on temporary array
	StreamCompaction::Efficient::memoryEfficientInclusiveScan(numRays, dev_compact_scan_array);

	// Step 3: inclusive scatter in place
	inclusiveScatterRays << <blocksPerGrid, blockSize >> >(dev_rayPool, dev_compact_tmp_array, dev_compact_scan_array, numRays);

	int last_index;
	cudaMemcpy(&last_index, dev_compact_scan_array + (numRays - 2), sizeof(int),
		cudaMemcpyDeviceToHost);

	int last_true_false;
	cudaMemcpy(&last_true_false, dev_compact_tmp_array + (numRays - 1), sizeof(int),
		cudaMemcpyDeviceToHost);

	return last_index + last_true_false;
}