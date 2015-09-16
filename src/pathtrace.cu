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

static Scene *hst_scene;
static glm::vec3 *dev_image;
static Geom *dev_geoms;
// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	//Copy geoms to dev_geoms
	
	int geoSize = hst_scene->geoms.size()*sizeof(Geom);
	Geom * hst_geoms = (Geom *)malloc(geoSize);

	std::copy(hst_scene->geoms.begin(),hst_scene->geoms.end(),hst_geoms);
	/* //??? or:
	hst_geoms = & hst_scene->geoms[0];
	*/

	cudaMalloc((void**)&dev_geoms, geoSize);
	cudaMemcpy(dev_geoms, hst_geoms, geoSize, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_geoms);
    cudaFree(dev_image);
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

	return ray_xy;
}

/**
* Test
* 1. Camera Generate Rays
*/
__global__ void Test(Camera cam, Geom * dev_geo, int geoNum, int iter, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);

		thrust::default_random_engine rng = random_engine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		Ray crntRay = GenerateRayFromCam(cam, x, y);
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
				temp_color = glm::vec3(1,0,0);
				break;
			case CUBE:
				temp_T = boxIntersectionTest(dev_geo[i], crntRay, temp_intrPoint, temp_intrNormal);
				temp_color = glm::vec3(0, 1, 0);
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
				pixelColor = temp_intrNormal;
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
__global__ void generateStaticDeleteMe(Camera cam, int iter, glm::vec3 *image) {
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
    //   * Compute one ray along each path - many will terminate.
    //     You'll have to decide how to represent your path rays and how
    //     you'll mark terminated rays.
    //   * Add all of the terminated rays' results into the appropriate pixels.
    //   * Stream compact away all of the terminated paths.
    //     You may use your implementation or `thrust::remove_if` or its
    //     cousins.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    // TODO: perform one iteration of path tracing
	int geoNum = hst_scene->geoms.size();
	Test <<<blocksPerGrid, blockSize >>>(cam, dev_geoms, geoNum, iter, dev_image);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
