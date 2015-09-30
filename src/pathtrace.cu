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
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/efficient.h"

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

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;

static Ray* dev_rays = NULL;
static Ray* dev_raysNew = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;

// Antialiasing
static int sampleTimes = 1;
static glm::vec3 *dev_image_antialias = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
    cudaMemset(dev_rays, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_raysNew, pixelcount * sizeof(Ray));
	cudaMemset(dev_raysNew, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_geoms, hst_scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, &(hst_scene->geoms)[0], hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, pixelcount * sizeof(Material));
	cudaMemcpy(dev_materials, &(hst_scene->materials)[0], hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_image_antialias, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_antialias, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_rays);
	cudaFree(dev_raysNew);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_image_antialias);

    checkCUDAError("pathtraceFree");
}


__global__ void initRays(Camera cam, int iter, Ray* rays, int sampleTimes) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		//thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_int_distribution<float> u01(0.0f, 1.0f);
		thrust::uniform_int_distribution<float> u04(-4.0f, 4.0f);
		float X, Y;
		glm::vec3 camRight = glm::cross(cam.up, cam.view);

		if(sampleTimes == 1){
			X = (-(cam.resolution.x / 2.0f - x ) * sin(cam.fov.x)) / cam.resolution.x * 2;
			Y = ((cam.resolution.y / 2.0f - y ) * sin(cam.fov.y)) / cam.resolution.y * 2;
		} else {
			X = (-(cam.resolution.x / 2.0f - x + u04(rng)) * sin(cam.fov.x)) / cam.resolution.x * 2;
			Y = ((cam.resolution.y / 2.0f - y + u04(rng)) * sin(cam.fov.y)) / cam.resolution.y * 2;
		}
		rays[index].direction = cam.view + X * camRight + Y * cam.up;
		rays[index].origin = cam.position;
		rays[index].color = glm::vec3(1.0f);
		rays[index].imageIndex = index;
		rays[index].run = true;
	}
}

__global__ void computeRays( Ray *rays, const Geom *geoms, const int objNumber) {
	int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	bool outsideFlag;
	float t = 0.0f;
	float closestT = 999999999999;
	glm::vec3 normal, intersectionPoint;

	for (int i = 0; i < objNumber; i++) {
		if (geoms[i].type == CUBE) {
			t = boxIntersectionTest(geoms[i], rays[index], intersectionPoint, normal, outsideFlag);
		}
		else if (geoms[i].type == SPHERE) {
			t = sphereIntersectionTest(geoms[i], rays[index], intersectionPoint, normal, outsideFlag);
		}
		if ( t > 0 && t < closestT ) {
			closestT = t;
			rays[index].hit = true;
			rays[index].intersectionGeomIndex = i;
			rays[index].intersectionPoint = intersectionPoint;
			rays[index].intersectionNormal = normal;
		}
	}
}

__global__ void fillImage(int frame, int frames, int iter, int depth, glm::vec3 *image, Ray *rays, const Geom *geoms, const Material *materials) {
	int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int imageIndex = rays[index].imageIndex;
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, imageIndex, depth);

	if ( !rays[index].hit ) {
		rays[index].run = false;
		//???
		//image[imageIndex] += rays[index].color;
	}
	else {
		int materialIndex = geoms[rays[index].intersectionGeomIndex].materialid;
		if (materials[materialIndex].emittance) {
			rays[index].run = false;
			image[imageIndex] += materials[materialIndex].color * materials[materialIndex].emittance * rays[index].color / (float)(frames + 1);
			//image[imageIndex] = image[imageIndex] * ((float)frame)/((float)frame+1) + rays[index].color * materials[materialIndex].color * materials[materialIndex].emittance / (float)(frame + 1);
		}
		else {
			scatterRay(rays[index], rays[index].color, rays[index].intersectionPoint, rays[index].intersectionNormal, materials[materialIndex], rng);
		}
	}
}


__global__ void averageImage( Camera cam, glm::vec3 *image_anti, glm::vec3 *image, int sampleTimes) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		image[index] = image_anti[index]/(float)sampleTimes;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int frames, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

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
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    // * Finally, handle all of the paths that still haven't terminated.
    //   (Easy way is to make them black or background-colored.)

    // calculate the object position according to frame number
    Geom *geoms = &(hst_scene->geoms)[0];
    glm::vec3 translationCurrent;
    bool blur = false;
    for(int i=0; i<hst_scene->geoms.size(); i++) {
    	if(geoms[i].moving) {
    		blur = true;
    		translationCurrent = geoms[i].translation + (geoms[i].translationGoal - geoms[i].translation) * ((float)frame/(float)frames) ;
    		geoms[i].transform = utilityCore::buildTransformationMatrix(translationCurrent, geoms[i].rotation, geoms[i].scale);
    		geoms[i].inverseTransform = glm::inverse(geoms[i].transform);
    		geoms[i].invTranspose = glm::inverseTranspose(geoms[i].transform);
    	}
    }
    if(blur) {
    	cudaMemcpy(dev_geoms, &(hst_scene->geoms)[0], hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    }

    int rayNumber = pixelcount;
    for(int i=0; i<sampleTimes; i++) {
    	initRays<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, dev_rays, sampleTimes);
    	checkCUDAError("initRays");

    	for(int d=0; d<traceDepth; d++) {
    		dim3 blocksPerGrid = (rayNumber + 64 - 1) / 64;

    		computeRays<<<blocksPerGrid, blockSize2d>>>( dev_rays, dev_geoms, hst_scene->geoms.size());
    		checkCUDAError("computeRays");

    		fillImage<<<blocksPerGrid, blockSize2d>>>(frame, frames, iter, d, dev_image_antialias, dev_rays, dev_geoms, dev_materials);
    		checkCUDAError("fillImage");

    		rayNumber = StreamCompaction::Efficient::compact(rayNumber, dev_raysNew, dev_rays);
    		//printf("ray number is: %d", rayNumber);
    		cudaMemcpy(dev_rays, dev_raysNew, pixelcount * sizeof(Ray), cudaMemcpyDeviceToDevice);
    	}
    }
    averageImage<<<blocksPerGrid2d, blockSize2d>>>(cam, dev_image_antialias, dev_image, sampleTimes);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
