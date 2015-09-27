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


#define ERRORCHECK 1
#define BLOCKSIZE 8

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
// TODO: static variables for device memory, scene/camera info, etc
// ...
static Camera *dev_cam = NULL;
static Ray *dev_rays = NULL; 
static Ray *dev_compaction = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    // TODO: initialize the above static variables added above
	cudaMalloc(&dev_cam, sizeof(Camera));
	cudaMemcpy(dev_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_geoms, hst_scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, hst_scene->geoms.data(), 
		hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, hst_scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, hst_scene->materials.data(), 
		hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
	cudaMemset(dev_rays, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_compaction, pixelcount * sizeof(Ray));
	cudaMemset(dev_compaction, 0, pixelcount * sizeof(Ray));
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_cam);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_rays);
	cudaFree(dev_compaction);
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

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
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

__global__ void initializeRay(Camera *cam, Ray *pathRays, int iter, glm::vec3 *img) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam->resolution.x && y < cam->resolution.y) {
        int index = x + (y * cam->resolution.x);

		if(iter == 1) {
			img[index] = glm::vec3(1.0f);
		}

		glm::vec3 vA = glm::cross(cam->view, cam->up);
		glm::vec3 vB = glm::normalize(glm::cross(vA, cam->view));
		glm::vec3 midPoint = cam->position + cam->view;

		glm::vec3 vH = vA * glm::length(cam->view) * atan(glm::radians(cam->fov.x));
		glm::vec3 vV = vB * glm::length(cam->view) * atan(glm::radians(cam->fov.y));

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		glm::vec3 pW = midPoint + vH * (2.0f * (float(x + u01(rng))/float(cam->resolution.x - 1.0f)) - 1) 
			+ vV * (1 - (2.0f * (float(y + u01(rng))/float(cam->resolution.y - 1.0f))));
	
		pathRays[index].origin = cam->position;
		pathRays[index].direction = glm::normalize(pW - cam->position);
		pathRays[index].color = glm::vec3(1.0f);
		pathRays[index].isOutside = true;
		pathRays[index].isTerminated = false;
		
	}

}

__global__ void trace_ray(Camera *cam, Ray* pathRays, Geom* geoms, Material* materials, 
	glm::vec3 *image, int iter, int geom_number, int depth, int maxDepth) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pixel_index = x + (y * cam->resolution.x);

	if(depth == maxDepth) {
		pathRays[pixel_index].isTerminated = true;
	}

	if (x < cam->resolution.x && y < cam->resolution.y) {
       
		int t = INT_MAX;
		glm::vec3 intersectionPoint, normal;
		glm::vec3 nearestIntersectionPoint, theNormal;
		Geom intersectedGeom;
		bool isOutside;
		bool intersected = false;

		for (int i = 0; i < geom_number; i++) {

			int currentT = t;
				//check if it intersected anything
				if (geoms[i].type == GeomType::SPHERE) {
					currentT = sphereIntersectionTest(geoms[i], pathRays[pixel_index], 
						intersectionPoint, normal, isOutside);
				} else if (geoms[i].type == GeomType::CUBE) {
					currentT = boxIntersectionTest(geoms[i], pathRays[pixel_index], 
						intersectionPoint, normal, isOutside);
				}

				if (currentT > 0 && currentT < t) {
					intersected = true;
					t = currentT;
					intersectedGeom = geoms[i];
					nearestIntersectionPoint = intersectionPoint;
					theNormal = normal;
				}
			}

			if (intersected) {
				//terminate if it's a light source
				if (materials[intersectedGeom.materialid].emittance > 0) {
						pathRays[pixel_index].isTerminated = true;
						image[pixel_index] += pathRays[pixel_index].color;
				} else {
					thrust::default_random_engine rng = makeSeededRandomEngine(iter, pixel_index, depth);
					scatterRay(pathRays[pixel_index], pathRays[pixel_index].color, nearestIntersectionPoint, 
								theNormal, materials[intersectedGeom.materialid], rng);
				}
			} else {
				pathRays[pixel_index].isTerminated = true;
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
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(BLOCKSIZE, BLOCKSIZE);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray is a (ray, color) pair, where color starts as the
    //     multiplicative identity, white = (1, 1, 1).
    //   * For debugging, you can output your ray directions as colors. */


	initializeRay<<<blocksPerGrid2d, blockSize2d>>>(dev_cam, dev_rays, iter, dev_image);

	checkCUDAError("initialize ray");
	
	int iterated_depth = 0;
	int alive_rays = pixelcount;

	for(int i = 0; i < traceDepth; i++) {
		trace_ray<<<blocksPerGrid2d, blockSize2d>>>(dev_cam, dev_rays, dev_geoms, 
			dev_materials, dev_image, iter, hst_scene->geoms.size(), i, traceDepth);

		//alive_rays = StreamCompaction::Efficient::rayCompact(alive_rays, dev_compaction, dev_rays);
		//cudaMemcpy(dev_rays, dev_compaction, alive_rays * sizeof(Ray), cudaMemcpyDeviceToDevice);

		//iterated_depth++;
	}
	
	checkCUDAError("trace ray");

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
