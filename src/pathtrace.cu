#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust\device_ptr.h>
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
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}
__device__ float rayIntersection(Geom geometry, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, int &materIdx, bool &outside)
{
	float temp_T = -1;
	switch (geometry.type)
	{
	case SPHERE:
		temp_T = sphereIntersectionTest(geometry, r, intersectionPoint, normal, outside);
		materIdx = geometry.materialid;
		break;
	case CUBE:
		temp_T = boxIntersectionTest(geometry, r, intersectionPoint, normal, outside);
		materIdx = geometry.materialid;// glm::vec3(0, 1, 0); 
		break;
	default:
		break;
	}
	return temp_T;
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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

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
static Ray *dev_ray = NULL;

static Ray *dev_compacted = NULL;
static Material *dev_m;
static Geom *dev_geo = NULL;
static Geom *dev_geoms;
static glm::vec3 *dev_test = NULL;
static glm::vec3 *dev_colormap = NULL;
static int * dev_comBool = NULL;
static int * dev_comResult = NULL;
static Ray *camera_ray = NULL;
static Ray *dev_compatedRay = NULL;
static bool * dev_terminate = NULL;


__global__ void SetDevRay(Camera cmr, Ray  *dev_ray, int iter){//Ray *dev_ray
	//camera_ray = new Ray();
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cmr.resolution.x && y < cmr.resolution.y) {
		int index = x + (y * cmr.resolution.x);
		glm::vec3 F = glm::normalize(cmr.view);
		glm::vec3 R = glm::normalize(glm::cross(F, cmr.up));
		glm::vec3 U = glm::normalize(glm::cross(R, F));
		float len = glm::length(cmr.view);
		int width = cmr.resolution.x;
		int height = cmr.resolution.y;
		float alpha = cmr.fov.y*PI / 180.f;
		glm::vec3 V = U*len *tan(alpha);
		float temp = width*1.0 / (height*1.0);
		glm::vec3 H = temp*glm::length(F)*R;
		float xx;
		float yy;

		//jittering rays
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 1);
		thrust::uniform_real_distribution<float> uee(-EPSILON, EPSILON);
		float offset = uee(rng);

		xx = 2.0* x / width - 1.0 + offset;
		yy = 1.0 - 2.0* y / height + offset;
		glm::vec3 point_pos = cmr.view + cmr.position + xx*H + yy*V;//glm::vec3 M_ = cmr.position + C_;
		//screen point to world point
		dev_ray[index].direction = glm::normalize(point_pos - cmr.position);
		dev_ray[index].origin = cmr.position;
		dev_ray[index].terminate = false;
		dev_ray[index].hitcolor = WHITE;
	}
}

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = (cam.resolution.x) * (cam.resolution.y);

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	// TODO: initialize the above static variables added above
	///////////////////

	int geoSize = hst_scene->geoms.size()*sizeof(Geom);
	Geom * hst_geoms = (Geom *)malloc(geoSize);
	std::copy(hst_scene->geoms.begin(), hst_scene->geoms.end(), hst_geoms);
	cudaMalloc((void**)&dev_geoms, geoSize);
	cudaMemcpy(dev_geoms, hst_geoms, geoSize, cudaMemcpyHostToDevice);

	//////////////
	///////
	//camera_ray = new Ray[pixelcount];

	int nG = hst_scene->geoms.size();
	int nM = hst_scene->materials.size();

	Material * mm = new Material[nM];
	Geom *gg = new Geom[nG];
	for (int i = 0; i < nM; i++){
		mm[i] = hst_scene->materials[i];
	}
	for (int i = 0; i < nG; i++){
		gg[i] = hst_scene->geoms[i];
	}
	cudaMalloc((void**)&dev_m, nM*sizeof(Material));//n*sizeof(M)
	cudaMemcpy(dev_m, mm, nM*sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_geo, nG*sizeof(Geom));
	cudaMemcpy(dev_geo, gg, nG*sizeof(Geom), cudaMemcpyHostToDevice);

	// TODO: initialize the above static variables added above
	cudaMalloc(&dev_ray, pixelcount * sizeof(Ray));
	cudaMemset(dev_ray, 0, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_colormap, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_colormap, 0, pixelcount * sizeof(glm::vec3));

	checkCUDAError("pathtraceInit");

}

__global__ void generateIamge(Camera cam, int iter, glm::vec3 *image, Ray *r) {

	/*int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	//if (id < (cam.resolution.x * cam.resolution.y)) {
	//if ()*/
	/*int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
	int id = x + (y * cam.resolution.x);
	if (r[id].terminate)
	image[id] += r[id].hitcolor;
	}*/
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < cam.resolution.x *cam.resolution.y) {

		if (r[id].terminate){
			image[id] += r[id].hitcolor;
			//r[id].hitcolor = glm::vec3();
		}
	}
}
__device__ void directlightcheking(Ray &r, Geom *dev_geom, int nG, const int light, Material m_light){
	Ray light_ray;
	light_ray.direction = r.origin - dev_geom[light].translation;
	light_ray.origin = r.origin;
	float t;
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	bool outside;
	for (int i = 0; i < nG; i++){
		if (i != light){
			if (dev_geom[i].type == SPHERE){
				t = sphereIntersectionTest(dev_geom[i], r, intersectionPoint, normal, outside);
			}
			if (dev_geom[i].type == CUBE){
				t = boxIntersectionTest(dev_geom[i], r, intersectionPoint, normal, outside);
			}
		}
		if (i == light)t = -1;
	}

	if (t < 0){
		r.hitcolor *= m_light.emittance*m_light.color;
	}
}
__global__ void raytracing(Ray *r, int CurrentRayNumber, Geom *dev_geom, int nG,int nM, Material *dev_m, int iter, int traced, int currentd)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < CurrentRayNumber){
		if (!r[id].terminate){
			glm::vec3 normal;
			glm::vec3 intersectionPoint;
			bool outside = false;
			float t = -2.0;
			int mId = 0;
			float mint = 30000;
			bool mark = false;
			Material m_light;
			int light;
			glm::vec3 F_normal;
			glm::vec3 F_intersectionPoint;
			bool F_outside = false;
			float F_t = -1.0;
			for (int i = 0; i < nM; i++){
				if (dev_m[i].emittance>0)//light
				{
					light = i;
					m_light = dev_m[i];
				}
			}

			for (int i = 0; i < nG; i++){

				if (dev_geom[i].type == SPHERE){
					t = sphereIntersectionTest(dev_geom[i], r[id], intersectionPoint, normal, outside);
				}
				if (dev_geom[i].type == CUBE){
					t = boxIntersectionTest(dev_geom[i], r[id], intersectionPoint, normal, outside);
				}
				if (t<0)continue;
				if (t > 0 && t < mint){//if I want to find the nearest intersect object
					mId = dev_geom[i].materialid;
					mint = t;
					mark = true;
					F_intersectionPoint = intersectionPoint;
					F_normal = normal;
					F_outside = outside;
					F_t = t;
			
					thrust::default_random_engine rng = makeSeededRandomEngine(iter, id, currentd);
					thrust::uniform_real_distribution<float> uee(0, EPSILON * 10);
					float offset = uee(rng);
					//along the normal
					F_intersectionPoint += offset*F_normal;
				}
			}
			if (F_t < 0)//hit nothing terminate
			{
				r[id].terminate = true;
				r[id].hitcolor *= BLACK;

			}
			else if (F_t > 0)//hit something
			{
				glm::vec3 emmited_c;
				glm::vec3 color;
				Material m = dev_m[mId];//if (mId<0 || mId>nM):erro
				if (m.emittance > 0){//***if hit light
					r[id].terminate = true;
					r[id].hitcolor *= m.emittance*m.color;
				}
				else{
					if (currentd == traced - 1){
						r[id].terminate = true;
						directlightcheking(r[id], dev_geom, nG, light, m_light);
					}
					thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);
					thrust::default_random_engine rng = makeSeededRandomEngine(iter, id, currentd);
			    	scatterRay(r[id], color, F_intersectionPoint, F_normal, emmited_c, m, rng);
					//if ((!m.hasReflective) && (!m.hasRefractive))//diffuse
					r[id].hitcolor *= m.color;
				}
			}
		}
	}
}

//input:ray

/*
for (int d = 0; d <= ilog2ceil(num) - 1; d++){
p1 = pow(2, d);
p2 = pow(2, d + 1);
Uscan << <1, num >> >(p1, p2, dev_boolb);//change end to n
}
put0 << <1, 1 >> >(dev_boolb, num);
output[n-1]=0;
//cudaMemcpy(&hst, &dev_idata[6], sizeof(int), cudaMemcpyDeviceToHost);
//std::cout << hst << "ss1";
for (int d = ilog2ceil(num) - 1; d >= 0; d--){
p1 = pow(2, d);
p2 = pow(2, d + 1);
Dscan << <1, num >> >(p1, p2, dev_boolb);
}*/
__global__ void mapBool(Ray * dev_ray, int *dev_bool, int currentmount){
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < currentmount){
		if (dev_ray[id].terminate)dev_bool[id] = 0;
		else dev_bool[id] = 1;
	}
}

__global__ void kernScatter(int n, Ray *odata, const Ray *idata, const int *bools, const int *indices)
{
	int k = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (bools[k] == 1){
		int t = indices[k];
		odata[t] = idata[k];
	}
}/*
void StreamCompation(int currentmount){
const dim3 Grid_128 = (currentmount + 128 - 1) / 128;
int p1, p2;
int last;
extern __shared__ int *dev_bool[];
extern __shared__ int *dev_bool_temp[];
mapBool << <Grid_128, 128 >> >(dev_ray, dev_bool, currentmount);
mapBool << <Grid_128, 128 >> >(dev_ray, dev_bool_temp, currentmount)
for (int d = 0; d <= utilityCore::ilog2ceil(currentmount) - 1; d++){
p1 = pow(2, d);
p2 = pow(2, d + 1);
Uscan << <Grid_128, 128 >> >(p1, p2, dev_bool_temp);//change end to n
}
put0 << <1, 1 >> >(dev_bool_temp, currentmount);
for (int d = utilityCore::ilog2ceil(currentmount) - 1; d >= 0; d--){
p1 = pow(2, d);
p2 = pow(2, d + 1);
Dscan <<<Grid_128, 128 >>>(p1, p2, dev_bool_temp);
}
cudaMemcpy(&last, &(dev_bool[currentmount - 1]), sizeof(int), cudaMemcpyDeviceToHost);
//cudaMalloc((void**)&dev_odata, last*sizeof(int));

kernScatter << <Grid_128, 128 >> >(last, dev_odata, dev_idata, dev_bool_temp, dev_bool);
//cudaMemcpy(dev_bool_temp
/*if (dev_bool[k] == 1){
int t = dev_boolb[k];//
odata[t] = idata[k];*/

//}*/
__global__ void EfficientScan(int * indata, int *outdata, int n){//current size
	//Example 39-2 in Gems
	extern __shared__ int temp[];//stores the updated bool

	//int id = (blockIdx.x * blockDim.x) + threadIdx.x;//tid
	int thid = threadIdx.x;
	int offset = 1;
	int blocks = blockIdx.x * blockDim.x * 2;

	temp[2 * thid] = indata[2 * thid + blocks];
	temp[2 * thid + 1] = indata[2 * thid + blocks * 2+ 1];
	//...do in block maybe..
//	int n = blockDim.x * 2;
	for (int d = n >> 1; d > 0; d >>= 1) //d = 0 to log2 n – 1
	{
		__syncthreads();
		if (thid < d){
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0)						//clear the last element
		temp[n - 1] = 0;

	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)  {
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	outdata[2*thid+blocks] = indata[2 * thid];
	outdata[2 * thid + blocks+1] = indata[2 * thid + 1];
}

__global__ void BlockSums(int n, int * newdata, const int *dev_outdata, const int *dev_indata) {
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	//last number of the 
	if (id < n){
		newdata[id] = dev_outdata[(id + 1) * n - 1] + dev_indata[(id + 1) * n - 1];
	}
}
__global__ void BlcockIncrement(int n, int *dev_data, const int *increments) {
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < n){
		dev_data[id] = dev_data[id] + increments[blockIdx.x];
	}
}



	void pathtrace(uchar4 *pbo, int frame, int iter) {
		const int traceDepth = hst_scene->state.traceDepth;
		const Camera &cam = hst_scene->state.camera;
		const int pixelcount = cam.resolution.x * cam.resolution.y;

		const dim3 blockSize2d(8, 8);
		const dim3 blocksPerGrid2d(
			(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
			(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

		//	const int &nG = hst_scene->geoms.size();
		//const int &nM = hst_scene->materials.size();

		const int &nG = hst_scene->geoms.size();
		const int &nM = hst_scene->materials.size();

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
		//}//use 256 warp erro


		//SetDevRay << <blocksPerGrid2d, blockSize2d >> >(cam, dev_ray, iter);
		const dim3 Grid_256 = (pixelcount + 256 - 1) / 256;
		const dim3 B_256 = 256;
		const dim3 Grid_128 = (pixelcount + 128 - 1) / 128;
		const dim3 B_128 = 128;
		int current_ray = pixelcount;
		for (int i = 0; i < traceDepth; i++)
		{//remove the ray number from pixel count,so the Grid size needs to be changed
			dim3 CurGrid_128 = (current_ray + 128 - 1) / 128;
			if (!i)
			{
				cudaMalloc(&dev_ray, current_ray*sizeof(Ray));
				SetDevRay << <blocksPerGrid2d, blockSize2d >> >(cam, dev_ray, iter);//shoot ray first time&one time
			}
			//(Ray *r, int CurrentRayNumber, Geom *dev_geom, int nG, int nM, Material *dev_m, int iter, int traced, int currentd)
			raytracing << <CurGrid_128, B_128 >> >(dev_ray, current_ray, dev_geo, nG,nM, dev_m, iter, traceDepth, i);
			checkCUDAError("raytrace");
			//**********streamCompaction***************//
			//*****************************************//
			//step1.compute temp bool ray
/*			cudaMalloc(&dev_comBool, current_ray * sizeof(int));
			mapBool << <CurGrid_128, B_128 >> >(dev_ray, dev_comBool, current_ray);
			int p0 = (current_ray + 128 - 1) / 128 / 128;
			while (p0 > 1){
				//step2.divide the array into blocks: CurGrid_128 * 128;128threads each block
				//step3.scan over each block
				EfficientScan << <CurGrid_128, B_128 >> >(dev_comBool, dev_comResult, current_ray);
				//step4.write total sum of each block to  a new array.
				//return the sum of each block

				dim3 gBlockSum = p0;
				BlockSums << <gBlockSum, B_128 >> >(p0, dev_blockSums, dev_comResult, dev_comBool);//39*128
				int p0 = p0 / 128;
			}
			BlcockIncrement << << CurGrid_128, B_128 >> > (current_ray, dev_comResult, dev_scannResult);
		
			//
			//step3.scatter

			kernScatter << <CurGrid_128, B_128 >> >(current_ray, dev_compatedRay, dev_ray, dev_comBool, dev_comResult);
			cudaMemcpy(dev_ray, dev_compacted, sizeof(Ray)*current_ray, cudaMemcpyDeviceToDevice);
			//free pointer so I can malloc again?
			cudaFree(dev_comBool);
			checkCUDAError("compact");
			//************************************/
			
		}
		generateIamge << <Grid_128, B_128 >> >(cam, iter, dev_image, dev_ray);
		cudaFree(dev_ray);
		///////////////////////////////////////////////////////////////////////////

		// Send results to OpenGL buffer for rendering
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

		// Retrieve image from GPU
		cudaMemcpy(hst_scene->state.image.data(), dev_image,
			pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

		checkCUDAError("pathtrace");
	}
	void pathtraceFree() {
		cudaFree(dev_image);  // no-op if dev_image is null
		// TODO: clean up the above static variables
		cudaFree(dev_ray);
		cudaFree(dev_compacted);
		cudaFree(dev_m);
		cudaFree(dev_geo);
		cudaFree(dev_colormap);

		checkCUDAError("pathtraceFree");
	}
