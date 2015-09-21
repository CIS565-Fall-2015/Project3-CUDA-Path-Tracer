#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust\device_ptr.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
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

static Scene *hst_scene= NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms;
static Material *dev_mats;
bool doStreamCompact = false;
Ray * dev_rays;
int ligntObjIdx = 0;


void pathtraceInit(Scene *scene,bool strCmpt) {
    hst_scene = scene;
	doStreamCompact = strCmpt;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
	ligntObjIdx = scene->lightIdx;

	//(1) Initialize array of path rays
	int raySize = pixelcount*sizeof(Ray);
	cudaMalloc((void**)&dev_rays, raySize);

	//Copy geoms to dev_geoms
	
	int geoSize = hst_scene->geoms.size()*sizeof(Geom);
	Geom * hst_geoms = (Geom *)malloc(geoSize);

	std::copy(hst_scene->geoms.begin(),hst_scene->geoms.end(),hst_geoms);
	/* //??? or:
	hst_geoms = & hst_scene->geoms[0];
	*/

	cudaMalloc((void**)&dev_geoms, geoSize);
	cudaMemcpy(dev_geoms, hst_geoms, geoSize, cudaMemcpyHostToDevice);

	//Copy materials to dev_mats
	int matSize = hst_scene->materials.size()*sizeof(Material);

	cudaMalloc((void**)&dev_mats, matSize);
	cudaMemcpy(dev_mats, hst_scene->materials.data(), matSize, cudaMemcpyHostToDevice);

	// dev_image initialize
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

	cudaFree(dev_rays);
	cudaFree(dev_geoms);
	cudaFree(dev_mats);
    cudaFree(dev_image);// no-op if dev_image is null

    checkCUDAError("pathtraceFree");
}

__device__ Ray GenerateRayFromCam(Camera cam, int x, int y)
{
	Ray ray_xy;
	ray_xy.origin = cam.position;

	glm::vec3 C_ = cam.view;
	glm::vec3 U_ = cam.up;
	glm::vec3 A_ = glm::cross(C_, U_);
	glm::vec3 B_ = glm::cross(A_, C_);
	glm::vec3 M_ = cam.position + C_;

	float tanPhi = tan(cam.fov.x*PI / 360);
	float tanTheta = tanPhi*(float)cam.resolution.x / (float)cam.resolution.y;
	glm::vec3 V_ = glm::normalize(B_)*glm::length(C_)*tanPhi;
	glm::vec3 H_ = glm::normalize(A_)*glm::length(C_)*tanTheta;

	float Sx = ((float)x + 0.5) / (cam.resolution.x - 1);
	float Sy = ((float)y + 0.5) / (cam.resolution.y - 1);
	glm::vec3 Pw = M_ + (2 * Sx - 1)*H_ - (2 * Sy - 1)*V_;
	glm::vec3 Dir_ = Pw - cam.position;

	ray_xy.direction = glm::normalize(Dir_);

	return ray_xy;
}

__global__ void kernInitPathRays(Camera cam,Ray * rays,int iter)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < cam.resolution.x && y < cam.resolution.y)
	{
		int index = x + (y * cam.resolution.x);
		rays[index].pixelIndex = glm::vec2(x,y);
		rays[index].imageIndex = index;
		rays[index].terminated = false;
		rays[index].origin = cam.position;
		rays[index].carry = glm::vec3(1,1,1);

		glm::vec3 C_ = cam.view;
		glm::vec3 U_ = cam.up;
		glm::vec3 A_ = glm::cross(C_, U_);
		glm::vec3 B_ = glm::cross(A_, C_);
		glm::vec3 M_ = cam.position + C_;

		float tanPhi = tan(cam.fov.x*PI / 180);
		float tanTheta = tanPhi*(float)cam.resolution.x / (float)cam.resolution.y;
		glm::vec3 V_ = glm::normalize(B_)*glm::length(C_)*tanPhi;
		glm::vec3 H_ = glm::normalize(A_)*glm::length(C_)*tanTheta;

		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);

		float Sx = ((float)x + u01(rng) - 0.5) / (cam.resolution.x - 1);
		float Sy = ((float)y + u01(rng) - 0.5) / (cam.resolution.y - 1);
		glm::vec3 Pw = M_ - (2 * Sx - 1)*H_ - (2 * Sy - 1)*V_;
		glm::vec3 Dir_ = Pw - cam.position;

		rays[index].direction = glm::normalize(Dir_);
		rays[index].lastObjIdx = -1;
		rays[index].origMatIdx = -1;
	}
}

__device__ float rayIntersection(Geom geometry, Ray r,glm::vec3& intersectionPoint, glm::vec3& normal, int &materIdx)
{
	float temp_T = -1;
	bool outside = true;//???????
	switch (geometry.type)
	{
	case SPHERE:
		temp_T = sphereIntersectionTest(geometry, r, intersectionPoint, normal,outside);
		materIdx = geometry.materialid;
		break;
	case CUBE:
		temp_T = boxIntersectionTest(geometry, r, intersectionPoint, normal, outside);
		materIdx = geometry.materialid;// glm::vec3(0, 1, 0);
		break;
	default:
		break;
	}
	return temp_T;
}

__global__ void kernComputeRay(int raysNum,Camera cam, Ray * rays, Material * dev_mat ,Geom * dev_geo, int geoNum,int iter,int depth)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int index = x + (y * cam.resolution.x);
	//if (x < cam.resolution.x && y < cam.resolution.y) 
	if (index<raysNum)
	{	
		if (rays[index].terminated)
		{
			return;
		}
		// intersection with objects
		glm::vec3 intrPoint;
		glm::vec3 intrNormal;
		float intrT = -1;
		int intrMatIdx;
		int intrObjIdx=-1;

		for (int i = 0; i<geoNum; i++)
		{
			glm::vec3 temp_intrPoint;
			glm::vec3 temp_intrNormal;
			float temp_T;
			int temp_MatIdx;

			temp_T = rayIntersection(dev_geo[i], rays[index], temp_intrPoint, temp_intrNormal, temp_MatIdx);
			if (temp_T < 0) continue;

			if (intrT < 0 || temp_T < intrT && temp_T >0)
			{
				intrT = temp_T;
				intrPoint = temp_intrPoint;
				intrNormal = temp_intrNormal;
				intrMatIdx = temp_MatIdx;
				intrObjIdx = i;
			}
		}
		if (intrT > 0)//intersect with obj, update ray
		{
			thrust::default_random_engine rr = makeSeededRandomEngine(iter, index, depth);
			scatterRay(rays[index], intrObjIdx, intrT, intrPoint, intrNormal, dev_mat[intrMatIdx], rr);
			rays[index].origMatIdx = intrMatIdx;
			rays[index].lastObjIdx = intrObjIdx;
		}
		else
		{
			rays[index].terminated = true;
			rays[index].carry = glm::vec3(0,0,0);// later background color
			rays[index].lastObjIdx = -1;
		}

	}
}

__global__ void kernUpdateImage(int raysNum,Camera cam, Ray * rays, glm::vec3 *image)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int index = x + (y * cam.resolution.x);
	if (index<raysNum)
	{
		if (rays[index].terminated)
		{
			image[rays[index].imageIndex] += rays[index].carry;
			rays[index].carry = glm::vec3(0,0,0);
		}
	}

}

__global__ void kernFinalImage(int iter, int raysNum, Camera cam, Ray * rays, glm::vec3 *image, Geom * dev_geo, Material * dev_mat,int geoNum, int lightIndex)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int index = x + (y * cam.resolution.x);
	if (index<raysNum)
	{	
		//Direct lighting
		//(1) random point on light
		// curently, only one box light source
		// !!!later : a.multiply lights; b.sphere light
		glm::vec4 pointOnLight(0,0,0,1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);
		thrust::uniform_real_distribution<float> u01(0, 1);
		pointOnLight.x = u01(rng)-0.5;
		pointOnLight.y = u01(rng)-0.5;
		pointOnLight.z = u01(rng)-0.5;
		pointOnLight = dev_geo[lightIndex].transform *pointOnLight; 
		//(2) surface point (ray.origin) to light_point, anything in between?
		glm::vec3 intrPoint;
		glm::vec3 intrNormal;
		float intrT = -1;
		int intrMatIdx;

		Ray surToLight;
		surToLight.origin = rays[index].origin;
		surToLight.direction = glm::normalize((glm::vec3)pointOnLight - rays[index].origin);
		//!!! later : Function this forloop into rayIntersection.
		for (int i = 0; i<geoNum; i++)
		{
			glm::vec3 temp_intrPoint;
			glm::vec3 temp_intrNormal;
			float temp_T;
			int temp_MatIdx;
			temp_T = rayIntersection(dev_geo[i], surToLight, temp_intrPoint, temp_intrNormal, temp_MatIdx);

			if (temp_T < 0) continue;
			if (intrT < 0 || temp_T < intrT && temp_T >0)
			{
				intrT = temp_T;
				intrPoint = temp_intrPoint;
				intrNormal = temp_intrNormal;
				intrMatIdx = temp_MatIdx;
			}
		}
		//(3) if nothing in between, cos ray, calc direct illumination
		if (intrMatIdx == lightIndex)
		{
			//Direct Illumination
			//!!! later : reduce bounce
			glm::vec3 color = dev_mat[lightIndex].emittance*dev_mat[lightIndex].color;
			color *= rays[index].carry;
			scatterRay(rays[index],-1, intrT, intrPoint, intrNormal, dev_mat[rays[index].origMatIdx], rng);
			color *= max(0.0f, glm::dot(glm::normalize(-rays[index].direction), glm::normalize(surToLight.direction)));
			image[rays[index].imageIndex] += color;
			rays[index].terminated = true;
		}
		else
		{
			image[rays[index].imageIndex] += glm::vec3(0, 0, 0); // !!! later background
			rays[index].terminated = true;
		}
	}
}

/**
 * Example function to generate static and test the CUDA-GL interop.
 * Delete this once you're done looking at it!
 */

struct is_terminated
{
	__host__ __device__ 
		bool operator()(const Ray ray_xy)
	{
		return ray_xy.terminated;
	}
};

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

	//(1) Initialize array of path rays
	kernInitPathRays<<<blocksPerGrid, blockSize>>>(cam, dev_rays,iter);

	//(2) For each depth:
	int geoNum = hst_scene->geoms.size();
	int totalRays = cam.resolution.x*cam.resolution.y;
	
	for (int i = 0; i < traceDepth; i++)
	{
		int bSize = blockSize.x*blockSize.y*blockSize.z;
		dim3 fullBlocksPerGrid((totalRays + bSize - 1) / bSize);
		thrust::device_ptr<Ray> RayStart(dev_rays);
		thrust::device_ptr<Ray> newRayEnd = RayStart + totalRays;
		// a. Compute one ray along each path
		kernComputeRay <<<fullBlocksPerGrid, bSize >>>(totalRays, cam, dev_rays, dev_mats, dev_geoms, geoNum, iter, i);
		// b. Add all terminated rays results into pixels
		kernUpdateImage <<<fullBlocksPerGrid, bSize >>>(totalRays, cam, dev_rays, dev_image);
		// c. Stream compact away/thrust::remove_if all terminated paths.
		if (doStreamCompact)
		{
			newRayEnd = thrust::remove_if(RayStart, newRayEnd, is_terminated());
			totalRays = (int)(newRayEnd - RayStart);
		}
	}
	//(3) Handle all not terminated
	int bSize = blockSize.x*blockSize.y*blockSize.z;
	dim3 fullBlocksPerGrid((totalRays + bSize - 1) / bSize);
	kernFinalImage <<<fullBlocksPerGrid, bSize >>>(iter,totalRays,cam, dev_rays, dev_image,dev_geoms,dev_mats,geoNum,ligntObjIdx);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}