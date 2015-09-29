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

#define BLOCK_DIM 8.0
#define BLOCK_SIZE (BLOCK_DIM*BLOCK_DIM)
static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
// TODO: static variables for device memory, scene/camera info, etc
// ...
static Geom *d_geom = NULL;
static Material *d_mat = NULL;
static RayInfo* d_rayInfo = NULL;
static RayInfo* d_tmp_rayInfo = NULL;
static int *d_sum = NULL;

static Texture *d_texture = NULL;
static std::vector<Texture> textureArr;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    // TODO: initialize the above static variables added above
	
	int geomSize = scene->geoms.size() * sizeof(Geom);
	cudaMalloc(&d_geom, geomSize);
	cudaMemcpy(d_geom, &(scene->geoms[0]), geomSize, cudaMemcpyHostToDevice);

	int matSize = scene->materials.size() * sizeof(Material);
	cudaMalloc(&d_mat, matSize);
	cudaMemcpy(d_mat, &(scene->materials[0]), matSize, cudaMemcpyHostToDevice);

	if (textureArr.size() == 0){
		for (int i = 0; i < scene->textures.size(); i++){
			int width = scene->textures[i].xSize;
			int height = scene->textures[i].ySize;
			glm::vec3 *d_pixels;
			cudaMalloc(&d_pixels, width * height * sizeof(glm::vec3));
			cudaMemcpy(d_pixels, &(scene->textures[i].pixels[0]), width * height * sizeof(glm::vec3), cudaMemcpyHostToDevice);

			Texture t;
			t.width = width;
			t.height = height;
			t.d_img = d_pixels;
			textureArr.push_back(t);
		}
	}
		
	int textureSize = textureArr.size() * sizeof(Texture);
	cudaMalloc(&d_texture, textureSize);
	cudaMemcpy(d_texture, &(textureArr[0]), textureSize, cudaMemcpyHostToDevice);

	cudaMalloc(&d_rayInfo, pixelcount * sizeof(RayInfo));
	cudaMalloc(&d_tmp_rayInfo, pixelcount * sizeof(RayInfo));
	cudaMalloc(&d_sum, pixelcount * sizeof(int));

	/*
	/////////////////////////////////////////////////////////////
	//test
	int sweepBlockSize = 3;
	int noSweepBlock = 3;

	char src[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' };
	int tmp[] = { 1, 0, 1, 1, 0, 0, 1, 0 };
	int IO[] = { 1, 0, 1, 1, 0, 0, 1, 0 };
	int n = 8;
	int *d_tmp;
	char *d_src;
	char *d_out;
	int *d_IO;
	cudaMalloc(&d_tmp, sizeof(int) * n);
	cudaMemcpy(d_tmp, tmp, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMalloc(&d_src, sizeof(char) * n);
	cudaMemcpy(d_src, src, sizeof(char) * n, cudaMemcpyHostToDevice);
	cudaMalloc(&d_IO, sizeof(int) * n);
	cudaMemcpy(d_IO, tmp, sizeof(int) * n, cudaMemcpyHostToDevice);

	calculateSum(d_tmp, n, sweepBlockSize, noSweepBlock);

	int outNo;
	cudaMemcpy(&outNo, &(d_tmp[n - 2]), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMalloc(&d_out, sizeof(char) * outNo);
	scatterIntTest<<<noSweepBlock, sweepBlockSize>>>(d_out, d_src, d_IO, d_tmp, n);

	char* out = new char[outNo];
	cudaMemcpy(out, d_out, sizeof(char) * outNo, cudaMemcpyDeviceToHost);

	for (int i = 0; i < outNo; i++)
		std::cout << out[i] << " ";
		*/
	/////////////////////////////////////////////////////////////
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    // TODO: clean up the above static variables
	cudaFree(d_geom);
	cudaFree(d_mat);
	cudaFree(d_rayInfo);
	cudaFree(d_tmp_rayInfo);
	cudaFree(d_sum);

	/*
	for (int i = 0; i < textureArr.size(); i++){
		cudaFree(textureArr[i].d_img);
	}
	*/
	cudaFree(d_texture);

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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	const dim3 blockSize2d(BLOCK_DIM, BLOCK_DIM);
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

    // TODO: perform one iteration of path tracing

	//Generate first set of ray based on the camera settings.
	int activeRayNo = pixelcount;

	glm::vec3 target = cam.view + cam.position;
	glm::vec3 side = glm::cross(cam.view, cam.up);
	glm::vec3 V = glm::normalize(cam.up) * tan(cam.fov.y / 2);
	glm::vec3 H = glm::normalize(side) * tan(cam.fov.x / 2);

	generateRays << <blocksPerGrid2d, blockSize2d >> >
		(d_rayInfo, cam.position, target, V, H, cam.resolution);

	//SHOOT THE RAYYY
	for (int i = 0; i < traceDepth; i++){
		//std::cout << activeRayNo << std::endl;

		if (activeRayNo > 0){
			int noBlock = ceil(activeRayNo / BLOCK_SIZE);
			iterate << <noBlock, BLOCK_SIZE >> >(d_rayInfo, dev_image, activeRayNo, i, iter,
				d_geom, hst_scene->geoms.size(), d_mat, d_texture);
			checkCUDAError("pathtraceInit");
			//steam compaction
			getOneZeroBit << <noBlock, BLOCK_SIZE, BLOCK_SIZE * sizeof(int) >> >
				(d_rayInfo, d_sum, activeRayNo);
			checkCUDAError("pathtraceInit");
			calculateSum(d_sum, activeRayNo, BLOCK_SIZE, noBlock);
			checkCUDAError("pathtraceInit");
			scatter << <noBlock, BLOCK_SIZE, BLOCK_SIZE * sizeof(int) >> >(d_tmp_rayInfo, d_rayInfo, d_sum, activeRayNo);
			checkCUDAError("pathtraceInit");
			cudaMemcpy(&activeRayNo, &(d_sum[activeRayNo - 2]), sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("pathtraceInit");
			RayInfo *tmp2 = d_rayInfo;
			d_rayInfo = d_tmp_rayInfo;
			d_tmp_rayInfo = tmp2;
		}

	}

	//TODO: terminates the rest of the rays.

	////////////////////////////////////////////////////////////////////////////

    //generateNoiseDeleteMe<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, dev_image);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

__global__ void generateRays(RayInfo* out_d_rays,
	glm::vec3 cam_pos, glm::vec3 cam_target,
	glm::vec3 V, glm::vec3 H, glm::vec2 resolution){

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = j * resolution.x + i;

	if (index < resolution.x * resolution.y){
		float NDC_X = 1 - (2 * i / resolution.x);
		float NDC_Y = 1 - (2 * j / resolution.y);

		glm::vec3 P = cam_target + V * NDC_Y + H * NDC_X;

		RayInfo info;
		info.ray.origin = cam_pos;
		info.ray.direction = glm::normalize(P - cam_pos);
		info.index = index;
		info.color = glm::vec3(1,1,1);

		out_d_rays[index] = info;
	}
}

__global__ void iterate(RayInfo* d_rayInfo, glm::vec3* d_image, int activeRayNo, int depth, int iter,
	Geom *d_geom, int numGeom, Material *d_mat, Texture *d_texture)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < activeRayNo && d_rayInfo[i].index != -1) {
		RayInfo r = d_rayInfo[i];
		float nearestT = -1;
		glm::vec3 nearestiPoint;
		glm::vec3 nearestNormal;
		glm::vec2 nearestUV;
		bool nearestOutside;
		int matId;

		for (int j = 0; j < numGeom; j++){
			Geom g = d_geom[j];
			float t = -1;
			glm::vec3 intersectionPoint;
			glm::vec3 normal;
			bool outside;
			glm::vec2 uv;
			
			if (g.type == GeomType::CUBE){
				t = boxIntersectionTest(g, r.ray, intersectionPoint, normal, outside);
			}
			else if (g.type == GeomType::SPHERE){
				t = sphereIntersectionTest(g, r.ray, intersectionPoint, normal, outside, uv);
			}

			if (t != -1 && (nearestT == -1 || t < nearestT)){
				nearestT = t;
				nearestiPoint = intersectionPoint;
				nearestNormal = normal;
				nearestOutside = outside;
				matId = g.materialid;
				nearestUV = uv;
			}
		}

		if (nearestT != -1){
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, i, depth);

			if (scatterRay(r.ray, r.color, nearestiPoint, nearestNormal, nearestOutside, d_mat[matId], rng, nearestUV, d_texture)){
				d_rayInfo[i].ray = r.ray;
				d_rayInfo[i].color = r.color;
				return;
			}
			//terminates! bump out the color
			d_image[r.index] += r.color;
			
		}
	
		//not intersecting anything, or terminated.
		d_rayInfo[i].index = -1;
	}
};

__global__ void getOneZeroBit(RayInfo* d_rayInfo, int* d_sum, int noRays){
	int ray_i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (ray_i < noRays){
		d_sum[ray_i] = d_rayInfo[ray_i].index == -1 ? 0 : 1;
	}
	else{
		d_sum[ray_i] = 0;
	}
}
	
__global__ void prescan(int* d_sum, int n, int originalSize) {
	extern __shared__ int sum[];

	int ray_i = (blockIdx.x * originalSize) + threadIdx.x;
	int i = threadIdx.x;

	if (ray_i < n && i < originalSize)
		sum[i] = d_sum[ray_i];
	else
		sum[i] = 0;


	//up-sweep
	for (unsigned int stride = 2; stride <= blockDim.x; stride *= 2){
		__syncthreads();
		if ((i + 1) % stride == 0){
			sum[i] += sum[i - (stride / 2)];
		}
	}

	if (i == blockDim.x - 1) sum[i] = 0;

	int tmpSum;
	//down-sweep
	for (unsigned int stride = blockDim.x; stride >= 2; stride /= 2){
		__syncthreads();
		if ((i + 1) % stride == 0){
			tmpSum = sum[i] + sum[i - (stride / 2)];
			sum[i - (stride / 2)] = sum[i];
			sum[i] = tmpSum;
		}
	}

	__syncthreads();

	if (i >= originalSize)
		return;

	if (i != blockDim.x - 1)
		d_sum[ray_i] = sum[i + 1];
	else
		d_sum[ray_i] += sum[i];
}

__global__ void offsetWithIncrement(int *d_sum, int n, int* d_increment, int sweepBlockSize){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n && blockIdx.x != 0){
		d_sum[i] += d_increment[blockIdx.x - 1];
	}
}


__global__ void getBlockIncrements(int* d_increment, int *d_sum, int noSweepBlock, int sweepBlockSize){

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < noSweepBlock){
		d_increment[i] = d_sum[sweepBlockSize * (i + 1) - 1];
	}
}

__global__ void scatter(RayInfo* out_d_rayInfo, RayInfo* d_rayInfo, int* d_sum, int rayNo){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i >= rayNo)
		return;

	if (d_rayInfo[i].index != -1){
		if (i == 0)
			out_d_rayInfo[0] = d_rayInfo[i];
		else 
			out_d_rayInfo[d_sum[i-1]] = d_rayInfo[i];
	}
}

/*
__global__ void scatterIntTest(char* out, char* in, int* IO, int* d_sum, int rayNo){
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i >= rayNo)
		return;

	if (IO[i] != 0){
		if (i == 0)
			out[0] = in[i];
		else
			out[d_sum[i - 1]] = in[i];
	}
}
*/

void calculateSum(int *d_sum, int n,
	int sweepBlockSize, int noSweepBlock){

	int expandedSize = pow(2,ceil(log2f(sweepBlockSize)));
	prescan << <noSweepBlock, expandedSize, expandedSize*sizeof(int) >> >(d_sum, n, sweepBlockSize);

	
	if (sweepBlockSize < n){
		//pad the array first! -> size must be a power of 2
		int* d_increment;
		int newNoBlock = ceil(noSweepBlock / (float)sweepBlockSize);
		cudaMalloc(&d_increment, sizeof(int) * noSweepBlock);
		getBlockIncrements << <newNoBlock, sweepBlockSize >> >(d_increment, d_sum, noSweepBlock, sweepBlockSize);

		calculateSum(d_increment, noSweepBlock, sweepBlockSize, newNoBlock);

		offsetWithIncrement << <noSweepBlock, sweepBlockSize >> >(d_sum, n, d_increment, sweepBlockSize);
		cudaFree(d_increment);
	}
}
