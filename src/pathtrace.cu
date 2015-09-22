#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

//#include <stream_compaction/efficient.h">

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define MAX_THREADS 512
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
//#define FILENAME "/d/Documents/cis565/hw3/test.txt"
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
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
static Ray* dev_out_rays;
//static glm::vec3* dev_final_colors;
static Geom* dev_geoms;
static Material* dev_materials;
//static glm::vec3* dev_colors;

//static BounceRay* dev_brays;
// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	const Geom* geoms = &(hst_scene->geoms)[0];
	const Material* materials = &(hst_scene->materials)[0];

	const int numObjects = hst_scene->geoms.size();
	cudaMalloc((void**)&dev_rays, pixelcount*sizeof(Ray));
	cudaMalloc((void**)&dev_out_rays, pixelcount*sizeof(Ray));
	cudaMalloc((void**)&dev_geoms, numObjects*sizeof(Geom));
	cudaMalloc((void**)&dev_materials, numObjects*sizeof(Material));

	cudaMemcpy(dev_geoms, geoms, numObjects*sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_materials, materials, numObjects*sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_rays);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
    checkCUDAError("pathtraceFree");
}

__global__ void initRays(int n, int iter, Camera cam, Ray* rays){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n){
		int x = index % cam.resolution.x;
		int y = index / cam.resolution.x;
		glm::vec3 left = glm::cross(cam.up, cam.view);

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.1, 0.1);

		float res2x = cam.resolution.x / 2.0f;
		float res2y = cam.resolution.y / 2.0f;

		float magx = -(res2x - x + u01(rng))*sin(cam.fov.x) / res2x;
		float magy = (res2y - y + u01(rng))*sin(cam.fov.y) / res2y;

		glm::vec3 direction = cam.view + magx*left + magy*cam.up;
		direction = glm::normalize(direction);

		rays[index].origin = cam.position;
		rays[index].direction = direction;
		rays[index].color = glm::vec3(1.0);
		rays[index].isAlive = true;
		rays[index].index = index;
	}
}

__global__ void intersect(int iter, int depth, int traceDepth, int n, Camera cam, Ray* rays, int numObjects, const Geom* geoms, const Material* materials){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	glm::vec3 normal;
	glm::vec3 intersectionPoint;
	float isIntersection;
	bool outside;
	
	glm::vec3 minNormal;
	glm::vec3 minIntersectionPoint;

	float minDist = INFINITY;
	int obj_index = -1;

	//if (blockIdx.x == 2429){
	//	printf("%d\n", blockIdx.x);
	//}

	//if (blockIdx.x == 4999){
	//	printf("%d\n",blockIdx.x);
	//}

	if (index < n){

		if (!rays[index].isAlive){
			return;
		}

		if (depth == traceDepth - 1 && rays[index].isAlive){
			rays[index].color = glm::vec3(0.0);
			rays[index].isAlive = false;
			return;
		}

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
		Ray ray = rays[index];

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
			scatterRay(rays[index], minIntersectionPoint, minNormal, materials[geoms[obj_index].materialid], rng);
		}
		else{
			rays[index].color = glm::vec3(0.0);
			rays[index].isAlive = false;
		}
	}
}

__global__ void updatePixels(int n, Ray* rays, glm::vec3* image){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n){
		if (!rays[index].isAlive){
			image[rays[index].index] += rays[index].color;
		}
	}
}

struct is_dead{
	__host__ __device__
	bool operator()(const Ray ray){
		return !ray.isAlive;
	}
};

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

	int numBlocks = (pixelcount-1) / MAX_THREADS + 1;

	initRays<<<numBlocks, MAX_THREADS>>>(pixelcount, iter, cam, dev_rays);
	checkCUDAError("initRays");

	int numAlive = pixelcount;
	Ray* last_ray;

	for (int d = 0; d < traceDepth; d++){
		numBlocks = (numAlive - 1) / MAX_THREADS + 1;
		//checkCUDAError("precheck");
		intersect<<<numBlocks, MAX_THREADS>>>(iter, d, traceDepth, numAlive, cam, dev_rays, numObjects, dev_geoms, dev_materials);
		//checkCUDAError("intersect");
		//updatePixels<<<numBlocks, MAX_THREADS>>>(numAlive, dev_rays, dev_image);

		//numAlive = StreamCompaction::Efficient::shared_compact(numAlive, dev_out_rays, dev_rays, is_dead());
		//numAlive = shared_compact(numAlive, dev_out_rays, dev_rays, is_dead());
		//cudaMemcpy(dev_rays, dev_out_rays, numAlive*sizeof(Ray), cudaMemcpyDeviceToDevice);

		//last_ray = thrust::remove_if(thrust::device, dev_rays, dev_rays + numAlive, is_dead());
		//numAlive = last_ray - dev_rays;
		if (numAlive == 0){
			break;
		}
	}

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}


/*
* Exclusive scan on idata, stores into odata, using shared memory
*/
__global__ void shared_scan(int n, int *odata, const int *idata){
	__shared__ int* temp;

	int index = (blockIdx.x * blockDim.x)+threadIdx.x;
	int offset = 1;

	temp[2 * index] = idata[2 * index];
	temp[2 * index + 1] = idata[2 * index + 1];

	for (int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if (index < d){
			int ai = offset*(2 * index + 1) - 1;
			int bi = offset*(2 * index + 2) - 1;
			temp[bi] += temp[ai];
		}
	}
	offset *= 2;
	if (index == 0){
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2){
		offset >>= 1;
		__syncthreads();
		if (index < d){
			int ai = offset*(2 * index + 1) - 1;
			int bi = offset*(2 * index + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	odata[2 * index] = temp[2 * index];
	odata[2 * index + 1] = temp[2 * index + 1];
}

template <typename T, typename Predicate> __global__ void kernMapToBoolean(int n, int* odata, T* idata, Predicate pred){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		odata[index] = pred(idata[index]);
	}
}

template <typename T> __global__ void kernScatter(int n, T* odata, T* idata, int* bools, int* scan){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		if (bools[index] == 1){
			odata[scan[index]] = idata[index];
		}
	}
}

template <typename T, typename Predicate> int shared_compact(int n, T* dev_odata, T* dev_idata, Predicate pred){
	// Returns the number of elements remaining, elements after the return value in odata are undefined
	// Assumes device memory
	int td = ilog2ceil(n);
	int n2 = (int)pow(2, td);

	int numBlocks = (n - 1) / MAX_THREADS + 1;
	int numBlocks2 = (n2 - 1) / MAX_THREADS + 1;
	int n_size = n * sizeof(int);
	int n2_size = n2 * sizeof(int);
	int out_size = 0;

	int* dev_temp;
	int* dev_temp_n2;
	int* dev_scan;

	cudaMalloc((void**)&dev_temp, n_size);
	cudaMalloc((void**)&dev_temp_n2, n2_size);
	cudaMalloc((void**)&dev_scan, n2_size);

	// Compute temp (binary)
	kernMapToBoolean << <numBlocks, MAX_THREADS >> >(n, dev_temp, dev_idata, pred);

	// Scan on temp
	cudaMemcpy(dev_temp_n2, dev_temp, n_size, cudaMemcpyDeviceToDevice); // Grow temp
	cudaMemset(dev_temp_n2 + n, 0, n2_size - n_size); // Pad with 0's
	shared_scan << <numBlocks2, MAX_THREADS >> >(n2, dev_scan, dev_temp_n2);

	// Scatter on scan
	kernScatter << <numBlocks, MAX_THREADS >> >(n, dev_odata, dev_idata, dev_temp, dev_scan);

	// Compute outsize
	int lastnum;
	int lastbool;
	cudaMemcpy(&lastnum, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastbool, dev_temp + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	out_size = lastnum + lastbool;
	return out_size;
}
