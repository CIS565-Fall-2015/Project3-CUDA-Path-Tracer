#include <cstdio>
#include <cuda.h>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
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

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
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
#endif //ERRORCHECK
}

__host__ __device__ thrust::default_random_engine random_engine(
        int iter, int index = 0, int depth = 0) {
    //return thrust::default_random_engine(utilhash((index + 1) * iter) ^ utilhash(depth));
    //return thrust::default_random_engine(utilhash(index ^ iter ^ depth));
    //return thrust::default_random_engine(utilhash(index + iter + depth));
//    return thrust::default_random_engine(utilhash(index) ^ utilhash(iter) ^ utilhash(depth));
    int h = utilhash((1 << 31) | ((depth + 5) << 22) | iter) ^ utilhash(index);
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

/* Static variables for device memory, scene/camera info, etc */
static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geom = NULL;
static Material *dev_mats = NULL;

static Pixel *dev_pixels = NULL;

/* Initialize static variables. */
void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_geom, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geom,  scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_mats, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_mats,  scene->materials.data(), scene->materials.size() *
            sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_pixels, pixelcount * sizeof(Pixel));
    cudaMemset(dev_pixels, 0, pixelcount * sizeof(Pixel));

    checkCUDAError("pathtraceInit");
}

/* Clean up static variables. */
void pathtraceFree() {
    // no-ops if pointers are null
    cudaFree(dev_image);
    cudaFree(dev_geom);
    cudaFree(dev_mats);
    cudaFree(dev_pixels);

    checkCUDAError("pathtraceFree");
}

__global__ void generateCameraRays(Camera cam, Pixel *pixels, int iter,
        glm::vec3 cam_right, glm::vec3 cam_up) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        // Jitter screen coordinates for anti-aliasing
        thrust::default_random_engine rng = random_engine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jit_x = x + u01(rng);
        float jit_y = y + u01(rng);

        float screen_x = -1 * (((float) jit_x * 2.f / (float)cam.resolution.x) - 1.f);
        float screen_y = -1 * (((float) jit_y * 2.f / (float)cam.resolution.y) - 1.f);

        glm::vec3 img_point = (cam.position + cam.view)
            + (cam_right * screen_x) + (cam_up * screen_y);
        glm::vec3 ray_dir = glm::normalize(img_point - cam.position);

        Ray r;
        r.origin = cam.position;
        r.direction = ray_dir;

        Pixel px;
        px.terminated = false;
        px.ray = r;
        px.color = glm::vec3(1, 1, 1);
        px.index = index;
        pixels[index] = px;
    }
}

__device__ float nearestIntersectionGeom(Ray r, Geom *geoms, int geomCount,
        Geom& nearest, glm::vec3 &intersection, glm::vec3 &normal) {
    float nearest_t = -1;
    for (int i = 0; i < geomCount; i++) {
        Geom g = geoms[i];
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

__global__ void intersect(Camera cam, glm::vec3 *image, Pixel *pixels,
        int depth, int iter,
        Geom *geoms, int geomCount, Material *mats) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (x < (cam.resolution.x * cam.resolution.y)) {
        Pixel px = pixels[x];
        int index = px.index;

        if (px.terminated == false) {
            glm::vec3 intersection = glm::vec3(0, 0, 0);
            glm::vec3 normal = glm::vec3(0, 0, 0);
            Geom nearest;
            float t = nearestIntersectionGeom(px.ray, geoms, geomCount, nearest,
                    intersection, normal);

            if (t > 0) {
                Material m = mats[nearest.materialid];
                thrust::default_random_engine rng = random_engine(iter, index, depth);
                scatterRay(pixels[index].ray, pixels[index].color,
                        intersection, normal, m, rng);

                if (m.emittance > 0) {
                    pixels[index].terminated = true;
                }

                // Debug
                //pixels[index].terminated = true;
                //pixels[index].color = glm::abs(pixels[index].ray.direction);
                //pixels[index].color = glm::abs(intersection/5.f);
            } else {
                pixels[index].color = glm::vec3(0, 0, 0);
                pixels[index].terminated = true;
            }
        }
    }
}

__global__ void debugCameraRays(Camera cam, glm::vec3 *image, Pixel *pixels) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);

        image[index] += pixels[index].ray.direction;
    }
}

__global__ void killNonterminatedRays(Camera cam, Pixel *pixels) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (x < (cam.resolution.x * cam.resolution.y)) {
        Pixel px = pixels[x];
        int index = px.index;
        if (px.terminated == false) {
            pixels[index].terminated = true;
            pixels[index].color = glm::vec3(0, 0, 0);
        }
    }
}

__global__ void storePixels(Camera cam, glm::vec3 *image, Pixel *pixels) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (x < (cam.resolution.x * cam.resolution.y)) {
        Pixel px = pixels[x];
        if (px.terminated == true) {
            int index = px.index;
            image[index] += px.color;
        }
    }
}

void shootCameraRays(Camera cam, Pixel *pixels, int iter,
        dim3 blockSize, dim3 blocksPerGrid) {
    float fovy = glm::radians(cam.fov.y);
    float aspectRatio = (float)cam.resolution.x / (float)cam.resolution.y;
    float tanPhi = glm::tan(fovy);
    float fovx = glm::atan(tanPhi * aspectRatio);
    float tanTheta = glm::tan(fovx);

    glm::vec3 cam_right = glm::cross(cam.view, cam.up) * tanTheta;
    glm::vec3 cam_up = glm::cross(cam_right, cam.view) * tanPhi;

    generateCameraRays<<<blocksPerGrid, blockSize>>>(cam, pixels, iter, cam_right, cam_up);
    checkCUDAError("shootCameraRays");
}

struct terminator {
    __device__ bool operator()(const Pixel px) {
        return px.terminated == true;
    }
};

int reapPixels(Camera cam, Pixel *pixels, int n) {
    Pixel *new_end = pixels + n;
    new_end = thrust::remove_if(thrust::device, pixels, pixels+n, terminator());
    return (new_end - pixels);
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

    shootCameraRays(cam, dev_pixels, iter, blockSize, blocksPerGrid);
    //debugCameraRays<<<blocksPerGrid, blockSize>>>(cam, dev_image, dev_pixels);

    int dBlockSize = 64;
    int dGridSize = (pixelcount + dBlockSize - 1) / dBlockSize;
    int count = pixelcount;

    for (int d = 0; d < traceDepth; d++) {
        intersect<<<dGridSize, dBlockSize>>>(
                cam, dev_image, dev_pixels,
                d, iter,
                dev_geom, hst_scene->geoms.size(), dev_mats);
        checkCUDAError("intersection");

//        storePixels<<<dGridSize, dBlockSize>>>(cam, dev_image, dev_pixels);
//        checkCUDAError("store");
//
//        count = reapPixels(cam, dev_pixels, count);
//        if (count == 0) { break; }
//        dGridSize = (count + dBlockSize - 1) / dBlockSize;
    }

    killNonterminatedRays<<<dGridSize, dBlockSize>>>(cam, dev_pixels);
    storePixels<<<dGridSize, dBlockSize>>>(cam, dev_image, dev_pixels);

    checkCUDAError("end");

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
