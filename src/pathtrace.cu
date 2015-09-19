#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/thrust.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
// TODO: Why did this start throwing a multi define error?
#define checkCUDAError(msg) checkCUDAErrorFn1(msg, FILENAME, __LINE__)
void checkCUDAErrorFn1(const char *msg, const char *file, int line) {
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
static Ray* dev_rays = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
	const Geom *geoms = &(hst_scene->geoms)[0];
	const Material *materials = &(hst_scene->materials)[0];
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	
	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
	cudaMemset(dev_rays, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_geoms, pixelcount * sizeof(Geom));
	cudaMemcpy(dev_geoms, geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, pixelcount * sizeof(Material));
	cudaMemcpy(dev_materials, materials, hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image); // no-op if dev_image is null
	cudaFree(dev_rays);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);

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

/**
 * Creates a ray throuhg each pixel on the screen.
 */
__global__ void InitializeRays(Camera cam, int iter, Ray* rays) {
	// will initialize rays based on camera positions
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x); // index in ray array

	// I don't think this check is actually necessary
	if (x < cam.resolution.x &&  y < cam.resolution.y) {
		glm::vec3 v = glm::cross(cam.up, cam.view);
		glm::vec3 direction;
		float half_res_x, half_res_y;
		float magnitude_x, magnitude_y;

		// Jitter within the pixel (one by one)
		thrust::default_random_engine rng = random_engine(iter, index, 0); // ray depth shouldn't matter here cause this is only called when depth is 0
		thrust::uniform_int_distribution<float> u01(-0.5f, 0.5f);

		half_res_x = cam.resolution.x / 2.0f;
		half_res_y = cam.resolution.y / 2.0f;
		magnitude_x = (-(half_res_x - x + u01(rng)) * sin(cam.fov.x)) / half_res_x;
		magnitude_y = ((half_res_y - y + u01(rng)) * sin(cam.fov.y)) / half_res_y;

		direction = cam.view + magnitude_x * v + magnitude_y * cam.up;

		rays[index].origin = cam.position;
		rays[index].direction = direction;
		rays[index].color = glm::vec3(0.0f);
		rays[index].pixel_index = index;
		rays[index].alive = true;
	}
}

__global__ void TraceBounce(Camera cam, int iter, int depth, glm::vec3 *image, Ray *rays, const Geom *geoms, const int numberOfObjects, const Material *materials) {
	// how to do the indexing on this? want it to be only for the remaining rays
	// I guess for now we just do the 1 through n thing where n is the number of rays left, and optimize the grid and block later
	// no this doesn't work, because then what pixel am i writing to?
	int index = threadIdx.x;
	int pixel_index = rays[index].pixel_index;
	float t = -1.0f;
	glm::vec3 color;
	glm::vec3 normal, intersectionPoint;
	thrust::default_random_engine rng = random_engine(iter, index, depth);

	// is a light a geom?
	for (int i = 0; i < numberOfObjects; i++) {
		int material_index = geoms[i].materialid;

		if (geoms[i].type == CUBE) {
			// ray march happens in the intersection tests, give the intersection point
			t = boxIntersectionTest(geoms[i], rays[index], intersectionPoint, normal);
		}
		else if (geoms[i].type == SPHERE) {
			t = sphereIntersectionTest(geoms[i], rays[index], intersectionPoint, normal);
		}
		else {
			//error. will add triangles later
		}

		// do i need to worry about how long along the line it is?
		if (t != -1.0f) {
			// two cases, either we hit a light, or we scatter again
			if (materials[material_index].emittance > EPSILON) {
				// if epsilon is greater than zero then we hit a light and we need to terminate
				rays[index].alive = false;
				//TODO: I am not sure this is the correct way to calculate color when hitting a light. pretty sure it isn't. What about specular?
				image[pixel_index] += rays[index].color * materials[material_index].color * materials[material_index].emittance;
				// maybe i only store the image here, as an addition, since it has been terminated. yeah
				// do the accume within an interation on the ray itself..

			}
			else {
				// not a light, do a bounce;
				scatterRay(rays[index], color, intersectionPoint, normal, materials[material_index], rng);
				// do we save color here or inside the scatter? ie should we modify rays color or the seperate color variable?

				//image[pixel_index] += color; //shouldn't be adding to this data structure i don't think
				// unless instead of this we save the color somewhere else (on the ray?)
				// need to update ray (it should be marked terminated inside the color update/scatter function)
				// ray should have been modified in place so no need to update it again here
			}
		}
		else {
			// case where nothing was hit and it hit the max distance it was allowed to travel? (so it won't be bouncing again right)
			// so we want to kill it and set it to the background color
			rays[index].alive = false;
			image[pixel_index] += glm::vec3(0.0f); // black in this case?
			// wait I am confused on this a bit.
			// we kill it and it didn't terminate, so should we be adding black? or setting to black?
			// adding cause we want to accumulate i guess just check this later
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
	const int numberOfObjects = hst_scene->geoms.size();
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
	//   * Compute one new (ray, color) pair along each path - note
	//     that many rays will terminate by hitting a light or nothing at all.
	//     You'll have to decide how to represent your path rays and how
	//     you'll mark terminated rays.
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


	InitializeRays<<<blocksPerGrid, blockSize>>>(cam, iter, dev_rays);

	//while the array size is not zero and less than depth
	int currentDepth = 0, threadsRemaining = pixelcount; //at first we will have one ray for each pixel
	while (threadsRemaining > 0 && currentDepth < traceDepth) {
		//trace bounce should have threadsremaining number of luanches
		TraceBounce<<<1, threadsRemaining>>>(cam, iter, currentDepth, dev_image, dev_rays, dev_geoms, numberOfObjects, dev_materials);
		threadsRemaining = StreamCompaction::Thrust::compact(threadsRemaining, dev_rays); //i don't want to use the otherone right now because it would have me copy to host and back to device. want to just keep on device
		currentDepth++;
	}

	// TODO: Remove this
    //generateNoiseDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);

	// need a device function here to change the ones that didn't finish to black (one's that remain)

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
