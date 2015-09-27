#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <thrust/device_vector.h>


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <stream_compaction/stream_compaction.h>

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
// TODO: static variables for device memory, scene/camera info, etc
// ...

static Path * dev_path;

//static Ray * dev_ray0;
//static Ray * dev_ray1;

//static thrust::device_vector<Ray> * dev_ray0;

//static Ray * dev_ray_cur;
//static Ray * dev_ray_next;

//static thrust::device_vector<Geom> dev_geom;			//global memory
//static thrust::device_vector<Material> dev_material;	//global
static Geom * dev_geom;
static Material * dev_material;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above

	

	cudaMalloc(&dev_path, pixelcount * sizeof(Path));

	

	cudaMalloc(&dev_geom, scene->geoms.size() * sizeof (Geom));
	cudaMemcpy(dev_geom, scene->geoms.data() , scene->geoms.size() * sizeof (Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_material,scene->geoms.size() * sizeof(Material));
	cudaMemcpy(dev_material,scene->materials.data() , scene->materials.size() * sizeof (Material), cudaMemcpyHostToDevice);

	//Geom * d = dev_geom;
	//for (std::vector<Geom>::iterator it = scene->geoms.begin() ; it != scene->geoms.end(); ++it) {
	//	int *src = &((*it)[0]);
	//	size_t sz = sizeof (Geom);

	//	cudaMemcpy(d, src, sizeof(int)*sz, cudaMemcpyHostToDevice);
	//	d += sz;
	//}
	


    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    // TODO: clean up the above static variables

	cudaFree(dev_path);

	cudaFree(dev_geom);
	cudaFree(dev_material);

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




__host__ __device__ void getCameraRayAtPixel(Path & path,const Camera &c, int x, int y,int iter,int index)
{
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);


	path.ray.origin = c.position;
	path.ray.direction = glm::normalize(c.view
		+ c.right * c.pixelLength.x * ((float)x - (float)c.resolution.x * 0.5f + u01(rng))  		//u01(rng) is for jiitering for antialiasing
		- c.up * c.pixelLength.y * ((float)y - (float)c.resolution.y * 0.5f + u01(rng)) 			//u01(rng) is for jiitering for antialiasing
		);

	if (c.lensRadiaus > 0)
	{
		//lens effect
		float r = c.lensRadiaus * u01(rng);
		float theta = u01(rng) * 2 * PI;

		
		float t = c.focalDistance * c.view.z / path.ray.direction.z;

		glm::vec3 pfocus = path.ray.origin + t * path.ray.direction;

		path.ray.origin = c.position + c.right * r * cos(theta) - c.up * r * sin(theta);
		path.ray.direction = glm::normalize(pfocus - path.ray.origin);
	}
	
	path.image_index = index;
	path.color = glm::vec3(1.0f);
	path.terminated = false;
	
}


/**
 * Generate Rays from camera through screen to the field
 * which is the first generation of rays
 *
 * Antialiasing - num of rays per pixel
 * motion blur - jitter scene position
 * lens effect - jitter camera position
 */
__global__ void generateRayFromCamera(Camera cam, int iter, Path* paths)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
		Path & path = paths[index];
		getCameraRayAtPixel(path,cam,x,y,iter,index);

		
       
    }
}



__global__ void pathTraceOneBounce(int iter, int depth,int num_paths,glm::vec3 * image
										,Path * paths
										,Geom * geoms, int geoms_size,Material * materials, int materials_size
										//,const thrust::device_vector<Geom> & geoms , const thrust::device_vector<Material> & materials
										)
{
	//int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	//int path_index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int path_index =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	
	if(path_index < num_paths)
	{
		Path & path = paths[path_index];
		//calculate intersection
		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		//naive parse through global geoms
		//for ( thrust::device_vector<Geom>::iterator it = geoms.begin(); it != geoms.end(); ++it)
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = false;
		for(int i = 0; i < geoms_size; i++)
		{
			//Geom & geom = static_cast<Geom>(*it);
			glm::vec3 tmp_intersect;
			glm::vec3 tmp_normal;
			Geom & geom = geoms[i];
			if( geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, path.ray, tmp_intersect, tmp_normal, outside);
			}
			else if( geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, path.ray, tmp_intersect, tmp_normal, outside);
			}
			else
			{
				//TODO: triangle
				//printf("ERROR: geom type error at %d\n",i);
				t = triangleIntersectionTest(geom, path.ray, tmp_intersect, tmp_normal, outside);
			}

			if(t > 0 && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}


		if(hit_geom_index == -1)
		{
			path.terminated = true;
			image[path.image_index] += BACKGROUND_COLOR;
		}
		else
		{
			//hit something
			Geom & geom = geoms[hit_geom_index];
			Material & material = materials[geom.materialid];


			//if (geom.type == TRIANGLE)
			//{
			//	path.terminated = true;
			//	image[path.image_index] += glm::vec3(1.0f);
			//	return;
			//}
			

			if (material.emittance > EPSILON)
			{
				//light source
				path.terminated = true;
				image[path.image_index] += path.color * material.color * material.emittance;
			}
			else
			{
				path.terminated = false;
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.image_index, depth);
				scatterRay(path.ray, path.color, intersect_point, normal, material, rng);
			}


			

		}
	}
}


struct is_path_terminated
{
  __host__ __device__
  bool operator()(const Path path)
  {
	  return path.terminated;
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
	const int blockSizeTotal = blockSideLength * blockSideLength;
    
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

    // TODO: perform one iteration of path tracing


    //generateNoiseDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);

	int depth = 0;

	generateRayFromCamera<<<blocksPerGrid2d,blockSize>>>(cam,iter,dev_path);
	checkCUDAError("generate camera ray");

	
	Path* dev_path_end = dev_path + pixelcount;
	int num_path = dev_path_end - dev_path;
	//loop
	while (dev_path_end != dev_path && depth < traceDepth)
	{
		
		dim3 blocksNeeded = (num_path + blockSizeTotal - 1) / blockSizeTotal ;
		pathTraceOneBounce<<<blocksNeeded,blockSize>>>(iter,depth, num_path  ,dev_image, dev_path
			, dev_geom, hst_scene->geoms.size(), dev_material, hst_scene->materials.size());
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth ++;

		//stream compaction
		dev_path_end = thrust::remove_if(thrust::device, dev_path, dev_path_end, is_path_terminated() );
		num_path = dev_path_end - dev_path;

		//TODO:self work efficient
		//num_path = StreamCompaction::Efficient::compact(num_path, dev_path);
		
		checkCUDAError("stream compaction");
	}

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
