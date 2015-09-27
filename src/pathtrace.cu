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

static Scene *hst_scene= NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms;
static Material *dev_mats;
bool doStreamCompact = false;
Ray * dev_rays;
Ray * dev_rays_temp;
int ttlLights = 0;
int * dev_lightIdxs;


void pathtraceInit(Scene *scene,bool strCmpt) {
    hst_scene = scene;
	doStreamCompact = strCmpt;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	//(1) Initialize array of path rays
	int raySize = pixelcount*sizeof(Ray);
	cudaMalloc((void**)&dev_rays, raySize);
	cudaMalloc((void**)&dev_rays_temp, raySize);
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

	//Copy lightIdxs to dev_lightIdxs
	ttlLights = hst_scene->lightIdxs.size();
	int lightSize = ttlLights *sizeof(int);
	cudaMalloc((void**)&dev_lightIdxs, lightSize);
	cudaMemcpy(dev_lightIdxs, hst_scene->lightIdxs.data(), lightSize, cudaMemcpyHostToDevice);

	// dev_image initialize
    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

	cudaFree(dev_rays);
	cudaFree(dev_rays_temp);
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

__device__ float rayIntersection(Geom geometry, Ray r,glm::vec3& intersectionPoint, glm::vec3& normal, int &materIdx,bool &outside)
{
	float temp_T = -1;
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
		bool intrOutside;

		for (int i = 0; i<geoNum; i++)
		{
			glm::vec3 temp_intrPoint;
			glm::vec3 temp_intrNormal;
			float temp_T;
			int temp_MatIdx;
			bool temp_outside;
			temp_T = rayIntersection(dev_geo[i], rays[index], temp_intrPoint, temp_intrNormal, temp_MatIdx, temp_outside);
			if (temp_T < 0) continue;

			if (intrT < 0 || temp_T < intrT && temp_T >0)
			{
				intrT = temp_T;
				intrPoint = temp_intrPoint;
				intrNormal = temp_intrNormal;
				intrMatIdx = temp_MatIdx;
				intrOutside = temp_outside;
			}
		}
		if (intrT > 0)//intersect with obj, update ray
		{
			thrust::default_random_engine rr = makeSeededRandomEngine(iter, index, depth);
			scatterRay(rays[index], intrOutside, intrT, intrPoint, intrNormal, dev_mat[intrMatIdx], rr);
			rays[index].origMatIdx = intrMatIdx;
			rays[index].lastObjIdx = intrOutside;
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

__global__ void kernFinalImage(int iter, int raysNum, Camera cam, Ray * rays, glm::vec3 *image, Geom * dev_geo, Material * dev_mat,int * dev_lightIdxs,int geoNum,int totalLights)
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
		glm::vec3 color(0, 0, 0);
		for (int i = 0; i < totalLights; i++)
		{
			int lightIndex = dev_lightIdxs[i];
			glm::vec4 pointOnLight(0, 0, 0, 1);
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);
			thrust::uniform_real_distribution<float> u01(0, 1);
			pointOnLight.x = u01(rng) - 0.5;
			pointOnLight.y = u01(rng) - 0.5;
			pointOnLight.z = u01(rng) - 0.5;
			pointOnLight = dev_geo[lightIndex].transform *pointOnLight;
			//(2) surface point (ray.origin) to light_point, anything in between?
			glm::vec3 intrPoint;
			glm::vec3 intrNormal;
			float intrT = -1;
			int intrMatIdx;
			bool intrOutside;

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
				bool temp_outside;
				temp_T = rayIntersection(dev_geo[i], surToLight, temp_intrPoint, temp_intrNormal, temp_MatIdx, temp_outside);

				if (temp_T < 0) continue;
				if (intrT < 0 || temp_T < intrT && temp_T >0)
				{
					intrT = temp_T;
					intrPoint = temp_intrPoint;
					intrNormal = temp_intrNormal;
					intrMatIdx = temp_MatIdx;
					intrOutside = temp_outside;
				}
			}
			//(3) if nothing in between, cos ray, calc direct illumination
			if (intrMatIdx == lightIndex)
			{
				//Direct Illumination
				//!!! later : reduce bounce
				color = dev_mat[lightIndex].emittance*dev_mat[lightIndex].color;
				color *= rays[index].carry;
				scatterRay(rays[index], intrOutside, intrT, intrPoint, intrNormal, dev_mat[rays[index].origMatIdx], rng);
				color *= max(0.0f, glm::dot(glm::normalize(-rays[index].direction), glm::normalize(surToLight.direction)));
			}
		}
		image[rays[index].imageIndex] += color;
		rays[index].terminated = true;
	}
}

/**
 * Example function to generate static and test the CUDA-GL interop.
 * Delete this once you're done looking at it!
 */
struct testdelete
{
	__host__ __device__
		bool operator()(const int a)
	{
		return a==1?false:true;
	}
};
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

__global__ void scan_sharedMem(int *dev_temp, int *Scan_odata)
{
	int n = blockDim.x * 2;

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int thid = threadIdx.x;

	Scan_odata[2 * index] = 2 * index;			//write scan results to device memory
	Scan_odata[2 * index + 1] = index;


	//Work-efficient scan with shared memory.
	//http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
	extern __shared__ int scan[];					//allocated on invocation
	int offset = 1;
	scan[2 * thid] = dev_temp[2 * index];				//loat ray.terminated to shared memory : scan
	scan[2 * thid + 1] = dev_temp[2 * index + 1];


	for (int d = n >> 1; d > 0; d >>= 1)	//build sum in place up the tree
	{
		__syncthreads();
		if (thid<d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;

			scan[bi] += scan[ai];
		}
		offset *= 2;

	}
	if (thid == 0)								//clear the last element
		scan[n - 1] = 0;

	for (int d = 1; d < n; d *= 2)			// traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;

			int t = scan[ai];
			scan[ai] = scan[bi];
			scan[bi] += t;
		}
	}
	__syncthreads();

	Scan_odata[2 * index] = scan[2 * thid];			//write scan results to device memory
	Scan_odata[2 * index + 1] = scan[2 * thid + 1];
}

__global__ void blockWise_sum(int *dev_temp,int *dev_scan,int * dev_bSum,int bSize)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int origIdx = (index + 1)*(bSize * 2) - 1;

	dev_bSum[index] = dev_temp[origIdx] + dev_scan[origIdx];
}

void delete_PrintIntArray(int * array, int length,std::string name)
{
	printf("%s = \n[",name);
	for (int i = 0; i < length; i++)
	{
		printf("%3d ", array[i]);
	}
	printf("]\n");
}
__global__ void sum_scan_incre(int * dev_scan,int*dev_incre,int t,int bSize)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int incrIdx = index / (bSize * 2);
	dev_scan[index] += dev_incre[incrIdx];
}
void ExclusiveScanTraverse(int * dev_inc,int&ttRayNum,int bSize,int *dev_temps,int * hst_a)
{
	int halfTtlRays = (int)((ttRayNum + 1) / 2);
	int GridSize = (halfTtlRays + bSize - 1) / bSize;

	if (GridSize == 1) //then start step 5
	{
		//dev_temps is dev_bIncre
		scan_sharedMem <<<GridSize, bSize, 2 * bSize*sizeof(int) >>>(dev_temps, dev_inc);
	}
	else
	{
		int *dev_scan;
		cudaMalloc((void**)&dev_scan, sizeof(int)*ttRayNum);
		scan_sharedMem << <GridSize, bSize, 2 * bSize*sizeof(int) >> >(dev_temps, dev_scan);

		int *dev_bSum;
		cudaMalloc((void**)&dev_bSum, sizeof(int)*GridSize);
		blockWise_sum << <(int)((GridSize + bSize - 1) / bSize), bSize >> >(dev_temps, dev_scan, dev_bSum, bSize);

		int *dev_incre;
		cudaMalloc((void**)&dev_incre, sizeof(int)*GridSize);
		ExclusiveScanTraverse(dev_incre, GridSize, bSize, dev_bSum, hst_a);
		
		GridSize = (ttRayNum + bSize - 1) / bSize;
		sum_scan_incre<<<GridSize,bSize>>>(dev_scan, dev_incre,ttRayNum,bSize);
		cudaMemcpy(hst_a, dev_scan, sizeof(int)*ttRayNum, cudaMemcpyDeviceToHost);
		//delete_PrintIntArray(hst_a, ttRayNum, "5. dev_scan");		//dev_incre
		
		cudaMemcpy(dev_inc, dev_scan, sizeof(int)*ttRayNum, cudaMemcpyDeviceToDevice);

		cudaFree(dev_incre);
		cudaFree(dev_bSum);
		cudaFree(dev_scan);
	}

}

__global__ void getUnterminatedTemp(Ray*ray,int*temp,int ttlRay)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//if (index < ttlRay)
		temp[index] = ray[index].terminated ? 0 : 1;
}

__global__ void streamCmp_scatter(Ray*irays, Ray*orays, int*tempBool, int * scanResult)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (!irays[index].terminated)
	{
		orays[scanResult[index]] = irays[index];
	}
}

void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	//(1) Initialize array of path rays
	kernInitPathRays <<<blocksPerGrid2d, blockSize2d >>>(cam, dev_rays, iter);

	//(2) For each depth:
	int geoNum = hst_scene->geoms.size();
	int totalRays = cam.resolution.x*cam.resolution.y;
	int why = totalRays;// totalRays;
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, 1, traceDepth);
	for (int i = 0; i < traceDepth; i++)
	{
		int bSize = 128;// blockSize.x*blockSize.y*blockSize.z;
		dim3 fullBlocksPerGrid((totalRays + bSize - 1) / bSize);
		// a. Compute one ray along each path
		kernComputeRay <<<fullBlocksPerGrid, bSize >>>(totalRays, cam, dev_rays, dev_mats, dev_geoms, geoNum, iter, i);
		// b. Add all terminated rays results into pixels
		kernUpdateImage <<<fullBlocksPerGrid, bSize >>>(totalRays, cam, dev_rays, dev_image);
		// c. Stream compact away/thrust::remove_if all terminated paths.

		if (doStreamCompact)
		{
			int ttRayNum = why;
			if (ttRayNum == 0) continue;
			
			bSize = 64;

			
			
			int GridSize = (ttRayNum + bSize - 1) / bSize;
			
			int *hst_temp = new int[ttRayNum];
			/*for (int i = 0; i < ttRayNum; i++)
			{
				thrust::uniform_real_distribution<float> u01(0, 1);
				
				if (u01(rng)>0.5)
				{
					hst_temp[i] = 1;
				}
				else hst_temp[i] = 0;
				
			}
			*/
			printf("\n\n/******* Test *******/\n");
			int *dev_temps;
			cudaMalloc((void**)&dev_temps, sizeof(int)*ttRayNum);
			getUnterminatedTemp<<<GridSize,bSize>>>(dev_rays, dev_temps, ttRayNum);
			
			//cudaMemcpy(dev_temps, hst_temp, sizeof(int)*ttRayNum, cudaMemcpyHostToDevice);
			cudaMemcpy( hst_temp,dev_temps, sizeof(int)*ttRayNum, cudaMemcpyDeviceToHost);
			//delete_PrintIntArray(hst_temp, ttRayNum, "1. original");		//dev_temp
			int lastInOrig;
			cudaMemcpy(&lastInOrig, dev_temps + ttRayNum - 1, sizeof(int), cudaMemcpyDeviceToHost);
			printf("lastInOrig:%d\n", lastInOrig);
			int *dev_temps_thrust;
			cudaMalloc((void**)&dev_temps_thrust, sizeof(int)*ttRayNum);
			getUnterminatedTemp << <GridSize, bSize >> >(dev_rays, dev_temps_thrust, ttRayNum);
			cudaMemcpy(dev_temps_thrust, hst_temp, sizeof(int)*ttRayNum, cudaMemcpyHostToDevice);
			checkCUDAError("000");

			int *dev_incre;
			cudaMalloc((void**)&dev_incre, sizeof(int)*ttRayNum);
			cudaMemset(dev_incre, 0, sizeof(int)*ttRayNum);
			checkCUDAError("111");
			printf("before scatter totalNum: %d\n", ttRayNum);
			ExclusiveScanTraverse(dev_incre,ttRayNum, bSize, dev_temps,hst_temp);
			checkCUDAError("aaa");
			cudaMemcpy(hst_temp, dev_incre, sizeof(int)*ttRayNum, cudaMemcpyDeviceToHost);
			//delete_PrintIntArray(hst_temp, ttRayNum, "..Final");		//dev_incre

			/*
			thrust::device_ptr<int> RayStart(dev_temps_thrust);
			thrust::device_ptr<int> newRayEnd = RayStart + ttRayNum;
			newRayEnd = thrust::remove_if(RayStart, newRayEnd, testdelete());
			int thrustTT = (int)(newRayEnd - RayStart);
			*/


			checkCUDAError("bbb");
			int lastInScan;
			cudaMemcpy(&lastInScan, dev_incre + ttRayNum - 1, sizeof(int), cudaMemcpyDeviceToHost);
			//*
			//thrust::remove_if stream compact:
			thrust::device_ptr<Ray> RayStart(dev_rays);
			thrust::device_ptr<Ray> newRayEnd = RayStart + ttRayNum;
			newRayEnd = thrust::remove_if(RayStart, newRayEnd, is_terminated());
			int thrustTT = (int)(newRayEnd - RayStart);
			//*/
			ttRayNum = lastInScan + lastInOrig;
			printf("after scatter totalNum: %d,\t thrust: %d\n\n", ttRayNum,thrustTT);
			why = ttRayNum;
			//totalRays = ttRayNum;
			//streamCmp_scatter<<<GridSize,bSize>>>(dev_rays, dev_rays_temp, dev_temps, dev_incre);		
			//checkCUDAError("bbb");
			//cudaMemcpy(dev_rays, dev_rays_temp, sizeof(Ray)*why, cudaMemcpyDeviceToDevice);
			//checkCUDAError("Copy Ray prob");
			//printf("");	
			//cudaMemcpy( hst_temp, dev_incre,sizeof(int)*ttRayNum, cudaMemcpyDeviceToHost);
			//delete_PrintIntArray(hst_temp, ttRayNum, "final. scan");		//dev_temp

			cudaFree(dev_incre);
			cudaFree(dev_temps);
		}
	}
	//(3) Handle all not terminated
	int bSize = 128;// blockSize.x*blockSize.y*blockSize.z;
	dim3 fullBlocksPerGrid((totalRays + bSize - 1) / bSize);
	kernFinalImage <<<fullBlocksPerGrid, bSize >>>(iter,totalRays,cam, dev_rays, dev_image,dev_geoms,dev_mats,dev_lightIdxs,geoNum,ttlLights);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}