#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

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
#define ERRORCHECK 1
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	+ cudaDeviceSynchronize();
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
__constant__ static glm::vec3 *dev_image = NULL;
__constant__ static Geom* dev_geoms = NULL;
__constant__ static Material* dev_materials = NULL;
static int geomcount = 0;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	Geom* hst_geoms = hst_scene->geoms.data();
	Material* hst_materials = hst_scene->materials.data();

	geomcount = hst_scene->geoms.size();

	cudaMalloc((void**)&dev_geoms, hst_scene->geoms.size()*sizeof(Geom));
	cudaMalloc((void**)&dev_materials, hst_scene->materials.size()*sizeof(Material));
	cudaMemcpy(dev_geoms, hst_geoms, hst_scene->geoms.size()*sizeof(Geom), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_materials, hst_materials, hst_scene->materials.size()*sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_geoms);
	cudaFree(dev_materials);

    checkCUDAError("pathtraceFree");
}

__global__ void initRayGrid(PathRay *oGrid, Camera cam){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathRay pr;
		pr.index = index;
		pr.color = glm::vec3(1.0f);

		pr.ray.origin = cam.position;
		pr.terminate = false;
		pr.matId = -1;

		// Grid center to pixel
		float pX = x - cam.resolution.x / 2;
		float pY = cam.resolution.y / 2 - y;

		// Vector: grid center to pixel
		glm::vec3 o2px = glm::vec3(cam.right.x*pX + cam.up.x*pY, cam.right.y*pX + cam.up.y*pY, cam.right.z*pX + cam.up.z*pY);
		// Ray vector
		pr.ray.direction = glm::vec3(cam.toGrid.x + o2px.x, cam.toGrid.y + o2px.y, cam.toGrid.z + o2px.z);

		oGrid[index] = pr;

		// Ray direction debug
		//float l = glm::length(ray.ray.direction);
		//image[index] += glm::vec3(abs(ray.ray.direction.x / l), abs(ray.ray.direction.y / l), 0);
	}
}


__global__ void interesect(PathRay *grid, Geom *iGeoms, Camera cam, const int grid_size, const int geomcount, glm::vec3 *image){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	// Memory access 2GB+
	// __shared__ memory variables can reduce by ~300MB
	// Dynamic blocksPerGrid can further reduce by ~700MB
	// Weird memory access violations and arbitrary incorrect materialid

	if (index < grid_size) {
		// Intersection test
		PathRay pr = grid[index];
		pr.hasIntersect = false;

		float oldLength = -1.0f;
		for (int i = 0; i < geomcount; ++i){
			float rayLength = 0.0f;
			glm::vec3 iPoint(0.0f);
			glm::vec3 iNormal(0.0f);
			bool outside = false;
			Geom g = iGeoms[i];
			if (g.type == SPHERE){
				rayLength = sphereIntersectionTest(g, pr.ray, iPoint, iNormal, outside);
			} else {
				rayLength = boxIntersectionTest(g, pr.ray, iPoint, iNormal, outside);
			}
			// Find the nearest intersection
			if (rayLength != -1.0f){
				pr.hasIntersect = true;
				if (oldLength == -1.0f || rayLength < oldLength){
					oldLength = rayLength;
					pr.intersect = iPoint;
					pr.normal = iNormal;
					pr.matId = g.materialid;
				}
			}
		}
		grid[index] = pr;
	}
};

__global__ void scatter(PathRay *grid, Material *iMaterials, Camera cam, const int grid_size, const int iter, const int depth){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	if (index < grid_size) {
		PathRay pr = grid[index];
		Material m = iMaterials[pr.matId];
		scatterRay(pr.ray, pr.color, pr.intersect, pr.ray.direction, pr.normal, m, random_engine(iter, index, depth));
		grid[index] = pr;
	}
};

__global__ void terminatePath(PathRay *grid, Material *iMaterials, Camera cam, const int grid_size){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	if (index < grid_size) {
		PathRay pr = grid[index];
		if (pr.hasIntersect){
			Material m = iMaterials[pr.matId];
			if (m.emittance > 0.0f){
				// Add to pixel
				pr.terminate = true;
				pr.color = pr.color * m.color * m.emittance;
			}
		}
		else {
			pr.terminate = true;
			pr.color = glm::vec3(0.0f);
		}
		grid[index] = pr;
	}
}

__global__ void fillPixel(PathRay *grid, glm::vec3 *image, Camera cam, const int grid_size){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	if (index < grid_size) {
		PathRay pr = grid[index];
		if (pr.terminate){
			image[pr.index] += pr.color;
		}
	}
}

__global__ void killPath(PathRay *grid, glm::vec3 *image, Camera cam, const int grid_size){
	// From camera as single point, to image grid with FOV
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = x + (y * cam.resolution.x);

	if (index < grid_size) {
		PathRay pr = grid[index];
		image[pr.index] += glm::vec3(0.0f);
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

    // Perform one iteration of path tracing

	thrust::device_vector<PathRay> dev_grid(pixelcount);
	PathRay *dev_grid_ptr = thrust::raw_pointer_cast(&dev_grid[0]);

	// initRayGrid
    initRayGrid<<<blocksPerGrid, blockSize>>>(dev_grid_ptr, cam);
	int grid_size = dev_grid.size();

	// For each traceDepth
	for (int d = 0; d < traceDepth; d++){
		// Intersection test
		interesect << <blocksPerGrid, blockSize, geomcount*sizeof(Geom) >> >(dev_grid_ptr, dev_geoms, cam, grid_size, geomcount, dev_image);
		checkCUDAError("intersect");

		// Mark all terminated paths
		terminatePath << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_materials, cam, grid_size);
		checkCUDAError("terminatePath");

		// Paint image
		fillPixel << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_image, cam, grid_size);
		checkCUDAError("fillPixel");

		// Stream compaction
		thrust::detail::normal_iterator<thrust::device_ptr<PathRay>> newGridEnd = thrust::remove_if(dev_grid.begin(), dev_grid.end(), is_terminated());
		checkCUDAError("compact");
		dev_grid.erase(newGridEnd, dev_grid.end());
		grid_size = dev_grid.size();

		// Scatter
		scatter << <blocksPerGrid, blockSize >> >(dev_grid_ptr, dev_materials, cam, grid_size, iter, d);
		checkCUDAError("scatter");
	}
	dev_grid.clear();

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
