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

static Scene *hst_scene= NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms;
static Material *dev_mats;
Ray * dev_rays;

// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

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

    // TODO: clean up the above static variables

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

	//??? something goes wrong with camera control left/right
	///!!!later:antialising
	return ray_xy;
}

__global__ void kernInitPathRays(Camera cam,Ray * rays)
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

		float tanPhi = tan(cam.fov.x*PI / 360);
		float tanTheta = tanPhi*(float)cam.resolution.x / (float)cam.resolution.y;
		glm::vec3 V_ = glm::normalize(B_)*glm::length(C_)*tanPhi;
		glm::vec3 H_ = glm::normalize(A_)*glm::length(C_)*tanTheta;

		float Sx = ((float)x + 0.5) / (cam.resolution.x - 1);
		float Sy = ((float)y + 0.5) / (cam.resolution.y - 1);
		glm::vec3 Pw = M_ + (2 * Sx - 1)*H_ - (2 * Sy - 1)*V_;
		glm::vec3 Dir_ = Pw - cam.position;

		rays[index].direction = glm::normalize(Dir_);
	}
}

__global__ void kernComputeRay(Camera cam, Ray * rays, Material * dev_mat ,Geom * dev_geo, int geoNum,int iter,int depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		int index = x + (y * cam.resolution.x);
		if (rays[index].terminated)
		{
			return;//!!! later compact
		}
		// intersection with objects
		glm::vec3 intrPoint;
		glm::vec3 intrNormal;
		float intrT = -1;
		//glm::vec3 pixelColor(0, 0, 0);
		Material intrMat;
		for (int i = 0; i<geoNum; i++)
		{
			glm::vec3 temp_intrPoint;
			glm::vec3 temp_intrNormal;
			float temp_T;
			Material temp_Mat;
			
			switch (dev_geo[i].type)
			{
			case SPHERE:
				temp_T = sphereIntersectionTest(dev_geo[i], rays[index], temp_intrPoint, temp_intrNormal);
				temp_Mat = dev_mat[dev_geo[i].materialid];
				break;
			case CUBE:
				temp_T = boxIntersectionTest(dev_geo[i], rays[index], temp_intrPoint, temp_intrNormal);
				temp_Mat = dev_mat[dev_geo[i].materialid];// glm::vec3(0, 1, 0);
				break;
			default:
				break;
			}
			if (temp_T < 0) continue;
			if (intrT < 0 || temp_T < intrT && temp_T >0)
			{
				intrT = temp_T;
				intrPoint = temp_intrPoint;
				intrNormal = temp_intrNormal;
				intrMat = temp_Mat;
			}
		}
		if (intrT > 0)//intersect with obj, update ray
		{
			if (intrMat.emittance>0)
			{
				rays[index].carry *= intrMat.emittance*intrMat.color;//???? is this right....?
				rays[index].terminated = true;
			}
			// Shading 
			else if (intrMat.hasReflective||intrMat.hasRefractive)
			{
				//!!! later : reflective or refractive
			}
			else if (intrMat.specular.exponent>0)
			{
				//!!! later : specular
			}
			//!!! later : scatter
			else // diffuse
			{//??? absorb
				thrust::default_random_engine rng = random_engine(index,iter,  depth);
				thrust::uniform_real_distribution<float> u01(0, 1);
				
				//if (u01(rng) > 0.4)
				{
					rays[index].origin = getPointOnRay(rays[index], intrT);
					thrust::default_random_engine rr = random_engine(iter, index, depth);//???!!! what's this....
					rays[index].direction = glm::normalize(calculateRandomDirectionInHemisphere(intrNormal, rr));
					rays[index].carry *= intrMat.color;// *0.6f;
				}
				/*else
				{
					rays[index].terminated = true;
					rays[index].carry = glm::vec3(0, 0, 0);// later background color
				}*/
				
			}
			
		}
		else
		{
			rays[index].terminated = true;
			rays[index].carry = glm::vec3(0,0,0);// later background color
		}

	}
}


__global__ void kernFinalImage(Camera cam, Ray * rays, glm::vec3 *image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y)
	{
		int index = x + (y * cam.resolution.x);
		if (rays[index].terminated)
		{
			image[index] += glm::vec3(0, 0, 0); // !!! later background
		}
	}

}

__global__ void kernUpdateImage(Camera cam, Ray * rays, glm::vec3 *image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) 
	{
		int index = x + (y * cam.resolution.x);	
		if (rays[index].terminated)
		{
			image[index] += rays[index].carry;
		}
	}

}

/**
* Test
* 1. Camera Generate Rays
*/
__global__ void Test(Camera cam, Ray * rays, Geom * dev_geo, Material * dev_mat, int geoNum, int iter, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);

		thrust::default_random_engine rng = random_engine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		Ray crntRay = rays[index];// GenerateRayFromCam(cam, x, y);
		/*
		//Ray Cast Direction test
		glm::vec3 DirColor = crntRay.direction;
		DirColor.x = DirColor.x < 0 ? (-DirColor.x) : DirColor.x;
		DirColor.y = DirColor.y < 0 ? (-DirColor.y) : DirColor.y;
		DirColor.z = DirColor.z < 0 ? (-DirColor.z) : DirColor.z;
		*/	
		glm::vec3 pixelColor(0, 0, 0);

		glm::vec3 intrPoint;
		glm::vec3 intrNormal;	
		float intrT = -1 ;

		// Intersection with objects/geoms
		//??? slow...
		for (int i = 0; i<geoNum; i++)
		{
			glm::vec3 temp_intrPoint;
			glm::vec3 temp_intrNormal;	
			float temp_T;
			glm::vec3 temp_color;
				
			switch (dev_geo[i].type)
			{
			case SPHERE:
				temp_T = sphereIntersectionTest(dev_geo[i], crntRay, temp_intrPoint, temp_intrNormal);
				temp_color = dev_mat[dev_geo[i].materialid].color;
				break;
			case CUBE:
				temp_T = boxIntersectionTest(dev_geo[i], crntRay, temp_intrPoint, temp_intrNormal);
				temp_color = dev_mat[dev_geo[i].materialid].color;// glm::vec3(0, 1, 0);
				break;
			default:
				break;
			}
			if (temp_T < 0) continue;
			if (intrT < 0 || temp_T < intrT)
			{ 
				intrT = temp_T; 
				intrPoint = temp_intrPoint;
				intrNormal = temp_intrNormal;
				pixelColor = temp_color;
			}
		}
		
		if (intrT>0)
			image[index] += pixelColor;
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

	//(1) Initialize array of path rays
	kernInitPathRays<<<blocksPerGrid, blockSize>>>(cam, dev_rays);

	//(2) For each depth:
	int geoNum = hst_scene->geoms.size();
	for (int i = 0; i < traceDepth; i++)
	{
		// a. Compute one ray along each path
		kernComputeRay <<<blocksPerGrid, blockSize >>>(cam, dev_rays, dev_mats, dev_geoms, geoNum, iter,i);
		// b. Add all terminated rays results into pixels
		kernUpdateImage<<<blocksPerGrid, blockSize >>>(cam,dev_rays,dev_image);
		// c. Stream compact away/thrust::remove_if all terminated paths.
		//thrust::remove_if(dev_rays[0],dev_rays+cam.resolution.x*cam.resolution.y,)
	}
	//(3) Handle all not terminated
	kernFinalImage<<<blocksPerGrid, blockSize >>>(cam,dev_rays,dev_image);//??? block size
	//Test <<<blocksPerGrid, blockSize >>>(cam,dev_rays, dev_geoms, dev_mats, geoNum, iter, dev_image);
	

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
