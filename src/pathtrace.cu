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

/* Static variables for device memory, scene/camera info, etc */
static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Ray *dev_rays = NULL;
static Geom *dev_geom = NULL;

/* Initialize static variables. */
void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
    cudaMemset(dev_rays, 0, pixelcount * sizeof(Ray));

    cudaMalloc(&dev_geom, pixelcount * sizeof(Ray));
    cudaMemcpy(dev_geom, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

/* Clean up static variables. */
void pathtraceFree() {
    // no-ops if pointers are null
    cudaFree(dev_image);
    cudaFree(dev_rays);
    cudaFree(dev_geom);

    checkCUDAError("pathtraceFree");
}

<<<<<<< HEAD
__global__ void generateCameraRays(Camera cam, Ray *rays) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        float screen_x = ((float) x * 2.f / (float)cam.resolution.x) - 1.f;
        float screen_y = -1 * (((float) y * 2.f / (float)cam.resolution.y) - 1.f);

        glm::vec3 cam_right = glm::cross(cam.view, cam.up);
        glm::vec3 cam_up = glm::cross(cam_right, cam.view);

        glm::vec3 img_point = (cam.position + cam.view)
            + (cam_right * screen_x) + (cam_up * screen_y);
        glm::vec3 ray_dir = glm::normalize(img_point - cam.position);

        Ray r;
        r.origin = cam.position;
        r.direction = ray_dir;
        rays[index] = r;
    }
}

__device__ float nearestIntersectionGeom(Ray r, Geom *geoms, int geomCount, Geom& nearest) {
    float nearest_t = -1;
    for (int i = 0; i < geomCount; i++) {
        Geom g = geoms[i];
        glm::vec3 intersection = glm::vec3(0, 0, 0);
        glm::vec3 normal = glm::vec3(0, 0, 0);
        float t = -1;

        switch (g.type) {
        case SPHERE:
            t = sphereIntersectionTest(g, r, intersection, normal);
            break;
        case CUBE:
            t = boxIntersectionTest(g, r, intersection, normal);
            break;
        }

        if (t > 0 && (t < nearest_t || nearest_t == -1)) {
            nearest = g;
            nearest_t = t;
        }
    }
    return nearest_t;
}

__global__ void findIntersections(int geomCount, Camera cam, Ray *rays,
        glm::vec3 *image, Geom *geoms) {
=======
/**
 * Example function to generate static and test the CUDA-GL interop.
 * Delete this once you're done looking at it!
 */
__global__ void generateNoiseDeleteMe(Camera cam, int iter, glm::vec3 *image) {
>>>>>>> 796cae6... generateStatic -> generateNoise
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        Ray r = rays[index];

        Geom nearest;
        float t = nearestIntersectionGeom(r, geoms, geomCount, nearest);

        if (t > 0) {
            image[index] += glm::vec3(1, 1, 1);
        } else {
            image[index] += glm::vec3(0, 0, 0);
        }
    }
}

__global__ void generateDebugCamera(Camera cam, Ray *rays, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        image[index] += rays[index].direction;
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

    generateCameraRays<<<blocksPerGrid, blockSize>>>(cam, dev_rays);
    checkCUDAError("rays");

    //generateDebugCamera<<<blocksPerGrid, blockSize>>>(cam, dev_rays, dev_image);

    findIntersections<<<blocksPerGrid, blockSize>>>(
            hst_scene->geoms.size(), cam, dev_rays, dev_image, dev_geom);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
