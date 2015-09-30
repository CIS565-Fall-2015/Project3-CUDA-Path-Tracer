#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <glm/gtc/matrix_inverse.hpp>

//#include <stream_compaction/efficient.h">

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define MAX_THREADS 64
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
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
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
static Geom* dev_geoms;
static Material* dev_materials;


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

__global__ void initRays(int n, int iter, Camera cam, Ray* rays, float cam_pos_dx, float cam_pos_dy){
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

		// Depth-of-field computations
		// Source : https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
		glm::vec3 focal_point = cam.position + cam.focal * cam.view;
		float t = -(glm::dot(cam.position, direction) + glm::dot(focal_point, cam.view)) / ((glm::dot(direction,cam.view))+0.000001);
		glm::vec3 intersection = cam.position + t*direction;

		//rays[index].origin = cam.position;
		rays[index].origin = cam.position + cam_pos_dx*left + cam_pos_dy*cam.up;
		direction = intersection - rays[index].origin;

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
			scatterRay(rays[index], minIntersectionPoint, minNormal, materials[geoms[obj_index].materialid], geoms[obj_index], rng);

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

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Initialize aperture for DOF
	int numBlocks = (pixelcount-1) / MAX_THREADS + 1;

	thrust::default_random_engine rng = makeSeededRandomEngine(iter, iter, 0);
	thrust::uniform_real_distribution<float> uAp(-cam.aperture, cam.aperture);

	float dx = uAp(rng);
	float dy = uAp(rng);

	// Initialize rays
	initRays<<<numBlocks, MAX_THREADS>>>(pixelcount, iter, cam, dev_rays, dx, dy);
	checkCUDAError("initRays");

	// Handle movement (TODO: make this faster?)
	Geom* geoms = &(hst_scene->geoms)[0];
	glm::vec3 new_translation;
	glm::vec3 new_direction;
	float dist;
	for (int i = 0; i < numObjects; i++){
		if (geoms[i].moving){
			new_direction = geoms[i].moveto - geoms[i].translation;
			dist = glm::length(new_direction);
			new_direction = glm::normalize(new_direction);
			thrust::uniform_real_distribution<float> uMove(0, dist);
			new_translation = geoms[i].translation + new_direction * uMove(rng);

			geoms[i].transform = utilityCore::buildTransformationMatrix(
				new_translation, geoms[i].rotation, geoms[i].scale);
			geoms[i].inverseTransform = glm::inverse(geoms[i].transform);
			geoms[i].invTranspose = glm::inverseTranspose(geoms[i].transform);
		}
	}
	cudaMemcpy(dev_geoms, geoms, numObjects*sizeof(Geom), cudaMemcpyHostToDevice);

	// Path tracing
	int numAlive = pixelcount;
	Ray* last_ray;

	for (int d = 0; d < traceDepth; d++){
		numBlocks = (numAlive - 1) / MAX_THREADS + 1;
		intersect<<<numBlocks, MAX_THREADS>>>(iter, d, traceDepth, numAlive, cam, dev_rays, numObjects, dev_geoms, dev_materials);
		updatePixels<<<numBlocks, MAX_THREADS>>>(numAlive, dev_rays, dev_image);

		numAlive = shared_compact(numAlive, dev_out_rays, dev_rays);
		cudaMemcpy(dev_rays, dev_out_rays, numAlive*sizeof(Ray), cudaMemcpyDeviceToDevice);

		//last_ray = thrust::remove_if(thrust::device, dev_rays, dev_rays + numAlive, is_dead());
		//numAlive = last_ray - dev_rays;
		if (numAlive == 0){
			break;
		}
	}

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

/*
* Exclusive scan on idata, stores into odata, using shared memory
*/
__global__ void kernSharedScan(int n, int *odata, const int *idata){
	extern __shared__ int temp[];

	int index = threadIdx.x;

	int offset = 1;

	temp[2 * index] = idata[2 * index + (blockIdx.x*blockDim.x*2)];
	temp[2 * index + 1] = idata[2 * index + 1 + (blockIdx.x*blockDim.x*2)];

	for (int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if (index < d){
			int ai = offset*(2 * index + 1) - 1;
			int bi = offset*(2 * index + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

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
	odata[2 * index + (blockIdx.x*blockDim.x*2)] = temp[2 * index];
	odata[2 * index + 1 + (blockIdx.x*blockDim.x*2)] = temp[2 * index + 1];
}

template <typename T> __global__ void kernFindAlive(int n, int* odata, T* idata){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		if (idata[index].isAlive){
			odata[index] = 1;
		}
		else {
			odata[index] = 0;
		}
		
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

__global__ void kernGetLastInBlocks(int n, int blockSize, int* dev_odata, int* dev_idata, int* dev_orig_idata){
	// n is the number of elements expected in the output array
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		dev_odata[index] = dev_idata[(index + 1)*blockSize - 1] + dev_orig_idata[(index + 1)*blockSize - 1];
	}
}

__global__ void kernInPlaceIncrementBlocks(int n, int* dev_data, int* increment){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		dev_data[index] = dev_data[index] + increment[blockIdx.x];
	}
}

void blockwise_scan(int n, int* dev_odata, int* dev_idata){
	int nfb = (n - 1) / MAX_THREADS + 1;
	int n2 = nfb * MAX_THREADS;
	int n_size = n * sizeof(int); // original input array size in bytes
	int fb_size = nfb*MAX_THREADS*sizeof(int); // full block size in bytes
	int shared_mem_size = MAX_THREADS * sizeof(int);

	int* dev_idata_fb;
	int* dev_odata_fb;

	//TODO: only need this is non-power-of-2
	cudaMalloc((void**)&dev_idata_fb, fb_size);
	cudaMalloc((void**)&dev_odata_fb, fb_size);

	cudaMemset(dev_idata_fb, 0, fb_size);
	cudaMemcpy(dev_idata_fb, dev_idata, n_size, cudaMemcpyDeviceToDevice);

	// Base case
	if (nfb == 1){
		kernSharedScan<<<nfb, MAX_THREADS/2, shared_mem_size>>>(MAX_THREADS, dev_odata_fb, dev_idata_fb);
		cudaMemcpy(dev_odata, dev_odata_fb, n_size, cudaMemcpyDeviceToDevice);
		cudaFree(dev_idata_fb);
		cudaFree(dev_odata_fb);
		return;
	}

	// Recurse
	int* dev_block_increments;
	int* dev_block_increments_scan;
	int numBlocks = (nfb - 1) / MAX_THREADS + 1;
	cudaMalloc((void**)&dev_block_increments, nfb*sizeof(int));
	cudaMalloc((void**)&dev_block_increments_scan, nfb*sizeof(int));

	kernSharedScan <<<nfb, MAX_THREADS/2, shared_mem_size>>>(MAX_THREADS, dev_odata_fb, dev_idata_fb);
	cudaDeviceSynchronize();

	kernGetLastInBlocks<<<numBlocks, MAX_THREADS>>>(nfb, MAX_THREADS, dev_block_increments, dev_odata_fb, dev_idata_fb);
	cudaDeviceSynchronize();

	blockwise_scan(nfb, dev_block_increments_scan, dev_block_increments);

	kernInPlaceIncrementBlocks<<<nfb, MAX_THREADS>>>(n2, dev_odata_fb, dev_block_increments_scan);

	cudaDeviceSynchronize();
	cudaMemcpy(dev_odata, dev_odata_fb, n_size, cudaMemcpyDeviceToDevice);

	cudaFree(dev_idata_fb);
	cudaFree(dev_odata_fb);
	cudaFree(dev_block_increments);
	cudaFree(dev_block_increments_scan);
}

template <typename T> int shared_compact(int n, T* dev_odata, T* dev_idata){
	// Returns the number of elements remaining, elements after the return value in odata are undefined
	// Assumes device memory

	int numBlocks = (n - 1) / MAX_THREADS + 1;
	int n_size = n * sizeof(int);
	int n2_size = numBlocks*MAX_THREADS*sizeof(int);
	int out_size = 0;

	int* dev_temp;
	int* dev_temp2;
	int* dev_scan;

	cudaMalloc((void**)&dev_temp, n_size);
	cudaMalloc((void**)&dev_temp2, n2_size);
	cudaMalloc((void**)&dev_scan, n2_size);

	// Compute temp (binary)
	kernFindAlive<<<numBlocks, MAX_THREADS>>>(n, dev_temp, dev_idata);
	cudaDeviceSynchronize();

	// Scan on temp
	blockwise_scan(n, dev_scan, dev_temp);

	// Scatter on scan
	kernScatter<<<numBlocks, MAX_THREADS>>>(n, dev_odata, dev_idata, dev_temp, dev_scan);

	// Compute outsize
	int lastnum;
	int lastbool;
	cudaMemcpy(&lastnum, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastbool, dev_temp + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	out_size = lastnum + lastbool;

	cudaFree(dev_temp);
	cudaFree(dev_temp2);
	cudaFree(dev_scan);
	return out_size;
}
