#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
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
        color.x = glm::clamp((int) (pix.x/iter* 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y/iter* 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z/iter* 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static glm::vec3 *cameraOrigin=NULL;
static glm::vec3 *cameraDirection=NULL;
static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static glm::vec3 *dev_origin,*dev_direction;
static glm::vec3 *dev_pixes,*pixes;
static glm::vec3 *dev_copyImage,*copyImage;
static bool *dev_inside,*inside;
static int *dev_terminated,*dev_pos,*pos,*terminated;
static Geom *geom,*dev_geom;
static Material *material,*dev_material;

// TODO: static variables for device memory, scene/camera info, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    // TODO: initialize the above static variables added above

	copyImage=new glm::vec3[pixelcount];
	inside=new bool[pixelcount];
	terminated=new int[pixelcount];
	pixes=new glm::vec3[pixelcount];
	pos=new int[pixelcount];

	cudaMalloc(&dev_copyImage, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_inside, pixelcount*sizeof(bool));
	cudaMalloc(&dev_terminated, pixelcount*sizeof(int));
	cudaMalloc(&dev_pixes, pixelcount*sizeof(glm::vec3));
	cudaMalloc(&dev_pos,pixelcount*sizeof(int));

	for(int i=0;i<pixelcount;++i) copyImage[i]=glm::vec3(1);
	for(int i=0;i<pixelcount;++i) inside[i]=false;
	for(int i=0;i<pixelcount;++i) terminated[i]=1;
	for(int i=0;i<pixelcount;++i) pixes[i]=glm::vec3(1);
	for(int i=0;i<pixelcount;++i) pos[i]=i;

	/*cudaMemcpy(dev_copyImage,copyImage,pixelcount*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inside,inside,pixelcount*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_terminated,terminated,pixelcount*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pixes,pixes,pixelcount*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pos,pos,pixelcount*sizeof(int),cudaMemcpyHostToDevice);
	*/
	initRay(cam);
	initGeom();
	initMaterial();
	camSetup(pixelcount);
    checkCUDAError("pathtraceInit");
}

kdtree *initTree(kdtree *root){
	//postorder method to first get the left and right child on GPU Memory, then replace it with the memory on CPU, then copy the whole point to GPU
	if(root==nullptr) return nullptr;
	kdtree *dev_lc=initTree(root->lc);
	kdtree *dev_rc=initTree(root->rc);
	kdtree *tmp(root);
	tmp->lc=dev_lc;
	tmp->rc=dev_rc;
	kdtree *dev_root;
	cudaMalloc(&dev_root,sizeof(kdtree));
	cudaMemcpy(dev_root,tmp,sizeof(kdtree),cudaMemcpyHostToDevice);
	return dev_root;
}

void initGeom(){
	geom=new Geom[hst_scene->geoms.size()];
	for(int i=0;i<hst_scene->geoms.size();++i){ 
		geom[i]=hst_scene->geoms[i];
		if(geom[i].mesh!=nullptr){
			glm::vec3 *dev_vertex;
			Mesh *dev_mesh,*mesh;
			int *dev_index;
			mesh=geom[i].mesh;
			cudaMalloc(&dev_vertex, sizeof(glm::vec3)*hst_scene->geoms[i].mesh->vertexNum);
			cudaMemcpy(dev_vertex,hst_scene->geoms[i].mesh->vertex,sizeof(glm::vec3)*hst_scene->geoms[i].mesh->vertexNum,cudaMemcpyHostToDevice);

			cudaMalloc(&dev_index, sizeof(int)*hst_scene->geoms[i].mesh->indexNum);
			cudaMemcpy(dev_index,hst_scene->geoms[i].mesh->indices,sizeof(int)*hst_scene->geoms[i].mesh->indexNum,cudaMemcpyHostToDevice);
			
			mesh->vertex=dev_vertex;
			mesh->indices=dev_index;
			
			mesh->tree=initTree(mesh->tree);
			cudaMalloc(&dev_mesh, sizeof(Mesh));
			cudaMemcpy(dev_mesh,mesh,sizeof(Mesh),cudaMemcpyHostToDevice);

			geom[i].mesh=dev_mesh;
		}
	}
	cudaMalloc(&dev_geom, hst_scene->geoms.size()*sizeof(Geom));
	cudaMemcpy(dev_geom, geom, hst_scene->geoms.size()*sizeof(Geom), cudaMemcpyHostToDevice);
}

void initMaterial(){
	material=new Material[hst_scene->materials.size()];
	for(int i=0;i<hst_scene->materials.size();++i) material[i]=hst_scene->materials[i];
	cudaMalloc(&dev_material, hst_scene->materials.size()*sizeof(Material));
	cudaMemcpy(dev_material, material, hst_scene->materials.size()*sizeof(Material), cudaMemcpyHostToDevice);
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    // TODO: clean up the above static variables
	cudaFree(cameraOrigin);
	cudaFree(cameraDirection);
	cudaFree(dev_origin);
	cudaFree(dev_direction);
	cudaFree(dev_geom);
	cudaFree(dev_material);
	cudaFree(dev_pixes);
	cudaFree(dev_inside);
	cudaFree(dev_terminated);
    checkCUDAError("pathtraceFree");
	cout<<"Memory is released"<<endl;
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

__device__ float findIntersection(Ray r, Geom *geom, int geomNum,int& interId,glm::vec3& interPos,glm::vec3& normal){
	float t=-1;
	for(int i=0;i<geomNum;++i){
		glm::vec3 tmpPos,tmpNormal;
		float tmp=-1;
		if(geom[i].type==CUBE){
			tmp=boxIntersectionTest(geom[i],r,tmpPos,tmpNormal);
		}
		else if(geom[i].type==SPHERE){
			tmp=sphereIntersectionTest(geom[i],r,tmpPos,tmpNormal);
		}
		else if(geom[i].type==MESH){
			tmp=meshIntersectionTest(geom[i],r,tmpPos,tmpNormal);
		}
		if(tmp!=-1&&(tmp<t||t==-1)){
			t=tmp;
			interPos=tmpPos;
			normal=tmpNormal;
			interId=i;
		}
	}
	return t;
}

__device__ float getRandomNum(int iter,int index,int depth){
	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::default_random_engine rng = random_engine(iter, index, depth);
	return u01(rng);
}

__device__ void reflectRay(Ray& r,glm::vec3 normal,glm::vec3 interPos){
	r.direction=r.direction-2.0f*normal*glm::dot(r.direction,normal);
	glm::normalize(r.direction);
	r.origin=interPos+0.001f*r.direction;
}

__device__ void refractRay(Ray& r,glm::vec3 normal,glm::vec3 interPos,float ref,float angle){
	glm::vec3 T=(-ref*glm::dot(normal,r.direction)-sqrt(angle))*normal+r.direction*ref;
	T=glm::normalize(T);
	r.direction=T;
	r.direction=glm::normalize(r.direction);
	r.origin=interPos+0.001f*r.direction;
}

__device__ void diffuseRay(Ray& r,glm::vec3 normal,glm::vec3 interPos,int iter,int index,int depth){
	thrust::default_random_engine rng = random_engine(iter, index, depth);
	r.direction=calculateRandomDirectionInHemisphere(normal,rng);
	r.direction=glm::normalize(r.direction);
	r.origin=interPos+0.001f*r.direction;
}

__device__ void specularRay(Ray& r,glm::vec3 normal,glm::vec3 interPos,float exponent,int iter,int index,int depth){
	thrust::default_random_engine rng = random_engine(iter, index, depth);
	r.direction=calculateRandomDirectionInHemisphereSpecular(normal,rng,exponent);
	r.direction=glm::normalize(r.direction);
	r.origin=interPos+0.001f*r.direction;
}

__device__ float getSchlickProb(float indexOfRefraction,glm::vec3 normal,glm::vec3 direction){
	float r0=pow((1.0f-indexOfRefraction)/(1.0f+indexOfRefraction),2);
	float th=glm::dot(normal,-direction);
	float refProb=r0+(1-r0)*pow(1-th,5);
	return refProb;
}

__device__ bool pathTraceThread(Ray& r,glm::vec3& color, Geom *geom, int geomNum, Material *material,int iter,int index,int depth,bool& inside){
	glm::vec3 interPos,normal;
	float t=INT_MAX;
	int interId=0;
	t=findIntersection(r,geom,geomNum,interId,interPos,normal);
	if(t>0){
		Material m=material[geom[interId].materialid];
		if(m.emittance>0){
			color=m.color*m.emittance;
			return true;
		}
		if(m.hasReflective){
			reflectRay(r,normal,interPos);
			return false;
		}
		else if(m.hasRefractive){//using fresenl's law, schlick's approach
			float refProb=getSchlickProb(m.indexOfRefraction,normal,r.direction);
			float ran=getRandomNum(iter,index+1,depth);
			if(ran<1.0f-refProb){
				float ref=m.indexOfRefraction;
				if(!inside) ref=1.0f/m.indexOfRefraction;
				float angle=1-ref*ref*(1-pow(glm::dot(normal,r.direction),2));
				if(angle<0){
					reflectRay(r,normal,interPos);
					return false;
				}
				else{
					inside=!inside;
					refractRay(r,normal,interPos,ref,angle);
					return false;
				}
			}
			else{
				reflectRay(r,normal,interPos);
				return false;
			}
		}
		else{
			float ran=getRandomNum(iter,index+1,depth);
			if(ran<0.5||m.specular.exponent==0){
				color=m.color;
				diffuseRay(r,normal,interPos,iter,index,depth);
				return false;
			}
			else{
				color=m.specular.color;
				specularRay(r,normal,interPos,m.specular.exponent,iter,index,depth);
				return false;
			}
		}
	}
	else{ 
		color=glm::vec3(0,0,0);
		return true;
	}
}

__global__ void pathTraceKernel(glm::vec3 *dev_origin,glm::vec3 *dev_direction,glm::vec3 *dev_pixes,bool *dev_inside,int *dev_terminated,
								Geom *geom,int geomNum,Material *material,int iter,int depth,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if (index<N) {
		Ray r;r.direction=dev_direction[index];r.origin=dev_origin[index];
		glm::vec3 result(1);
		bool inside=dev_inside[index];
		/*for(int i=0;i<=depth;++i){
			if(i==depth) result=glm::vec3(0,0,0);
			else{
				glm::vec3 color(1);
				bool end=pathTraceThread(r,color,geom,geomNum,material,iter,index,i,inside);
				result=result*color;
				if(end) break;
			}
		}*/
		glm::vec3 color(1);
		bool end=pathTraceThread(r,color,geom,geomNum,material,iter,index,depth,inside);
		result=result*color;
		dev_direction[index]=r.direction;
		dev_origin[index]=r.origin;
		dev_inside[index]=inside;
		dev_terminated[index]=!end;
		dev_pixes[index]*=result;
	}
}

__global__ void testCamSetupKernel(Camera cam, glm::vec3 *dev_cam, glm::vec3 *image){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
		image[index].x+=abs(dev_cam[index].x);
		image[index].y+=abs(dev_cam[index].y);
		image[index].z+=abs(dev_cam[index].z);
	}
}

void initRay(const Camera &cam){
	//course note from cis560
	glm::vec3 A,B,M,H,V;
	cameraOrigin=new glm::vec3[cam.resolution.x*cam.resolution.y];
	cameraDirection=new glm::vec3[cam.resolution.x*cam.resolution.y];
	A=glm::cross(cam.view,cam.up);
	B=glm::cross(A,cam.view);
	M=cam.position+cam.view;
	H=tan(1.0f*cam.fov.x/180.0f)*glm::length(cam.view)*glm::normalize(A);
	V=tan(1.0f*cam.fov.y/180.0f)*glm::length(cam.view)*glm::normalize(B);
	for(int i=0;i<cam.resolution.y;++i){
		for(int j=0;j<cam.resolution.x;++j){
			float sx,sy;
			sx=(1.0*j)/cam.resolution.x;
			sy=(1.0*i)/cam.resolution.y;
			glm::vec3 R=M+(1-2*sx)*H+(1-2*sy)*V;
			cameraOrigin[i*cam.resolution.x+j]=cam.position;
			cameraDirection[i*cam.resolution.x+j]=glm::normalize(R-cam.position);
		}
	}
}

void camSetup(int imageCount){
	cudaMalloc(&dev_origin, imageCount*sizeof(glm::vec3));
	cudaMemcpy(dev_origin, cameraOrigin, imageCount* sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_direction, imageCount*sizeof(glm::vec3));
	cudaMemcpy(dev_direction, cameraDirection, imageCount* sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

__global__ void test(glm::vec3 *pixes, glm::vec3 * copyImage,int *terminated,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(terminated[index]==1)
			copyImage[index]*=pixes[index];
		else{ 
			if(pixes[index]==glm::vec3(0))
				copyImage[index]=glm::vec3(0);
			else{
				copyImage[index]*=pixes[index];
				pixes[index]=glm::vec3(1);
			}
		}
	}
}

__global__ void test1(glm::vec3 * copyImage,int *terminated,int *pos,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		copyImage[pos[index]]=glm::vec3(0);
	}
}


__global__ void copyImageToRecord(glm::vec3 *image,glm::vec3 *copy,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		image[index]+=copy[index];
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

    const int blockSideLength =20;
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

    //generateNoiseDeleteMe<<<blocksPerGrid, blockSize>>>(cam, iter, dev_image);
	//testCamSetupKernel<<<blocksPerGrid,blockSize>>>(cam,dev_cam,dev_image);
    ///////////////////////////////////////////////////////////////////////////
	int geomNum=hst_scene->geoms.size();
	int matNum=hst_scene->materials.size();
	int depth=hst_scene->state.traceDepth;
	int remain=cam.resolution.x*cam.resolution.y;

	cudaMemcpy(dev_origin, cameraOrigin, remain* sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_direction, cameraDirection, remain* sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_copyImage,copyImage,pixelcount*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_inside,inside,pixelcount*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_terminated,terminated,pixelcount*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pixes,pixes,pixelcount*sizeof(glm::vec3),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pos,pos,pixelcount*sizeof(int),cudaMemcpyHostToDevice);

	for(int i=0;i<depth;++i){
		int bs=128;
		dim3 gs((remain+bs-1)/bs);
		pathTraceKernel<<<gs, bs>>>(dev_origin, dev_direction, dev_pixes,dev_inside,dev_terminated,dev_geom, geomNum, dev_material,iter,i,remain);
		int *idata=new int[remain];
		cudaMemcpy(idata,dev_terminated,remain*sizeof(int),cudaMemcpyDeviceToHost);
		remain=compact(remain,dev_origin,dev_direction,dev_pos,dev_pixes,dev_copyImage,dev_inside,idata,bs);
		delete(idata);
		cudaMemcpy(dev_terminated,terminated,remain*sizeof(int),cudaMemcpyHostToDevice);
		//test<<<gs,bs>>>(dev_pixes,dev_copyImage,dev_terminated,remain);
	}
	test1<<<(remain+127)/128,128>>>(dev_copyImage,dev_terminated,dev_pos,remain);
	copyImageToRecord<<<(cam.resolution.x*cam.resolution.y+255)/256,256>>>(dev_image,dev_copyImage,cam.resolution.x*cam.resolution.y);
	//getchar();
    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid, blockSize>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtrace");
	//cout<<"one iter"<<endl;
}


/*
stream compaction
*/

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}

__global__ void upSwapOnGPU(int *idata,int step,int n,int newN){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<newN){
		if(step==1&&index>=n) idata[index]=0;
		if((index+1)%(step*2)==0) idata[index]+=idata[index-step];
	}
}

__global__ void downSwapOnGPU(int *idata,int step,int n,int newN){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<newN){
		if(step*2==newN&&index==newN-1) idata[index]=0;
		if((index+1)%(step*2)==0){
			int tmp=idata[index-step];
			idata[index-step]=idata[index];
			idata[index]+=tmp;
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata,int blockSize) {
	int newN=pow(2,ilog2ceil(n));
	int *dev_idata;
	cudaMalloc((void**)&dev_idata, newN * sizeof(int));
	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockPerGrid=((newN+blockSize-1)/blockSize);
	
	int step=1;
	while(step<newN){
		upSwapOnGPU<<<blockPerGrid,blockSize>>>(dev_idata,step,n,newN);
		step*=2;
	}
	step/=2;
	while(step!=0){
		downSwapOnGPU<<<blockPerGrid,blockSize>>>(dev_idata,step,n,newN);
		step/=2;
	}
	
	cudaMemcpy(odata,dev_idata,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dev_idata);
}

__global__ void compactOrigin(int *idata,int *scanned, glm::vec3 *origin,glm::vec3 *origin_copy,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(idata[index]==1){
			origin[scanned[index]]=origin_copy[index];
		}
	}
}

__global__ void compactDirection(int *idata,int *scanned, glm::vec3 *direction,glm::vec3 *direction_copy,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(idata[index]==1){
			direction[scanned[index]]=direction_copy[index];
		}
	}
}

__global__ void compactInside(int *idata,int *scanned, bool *inside,bool *inside_copy,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(idata[index]==1){
			inside[scanned[index]]=inside_copy[index];
		}
	}
}

__global__ void compactPixes(int *idata,int *scanned, glm::vec3 *pixes,glm::vec3 *pixes_copy,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(idata[index]==1){
			pixes[scanned[index]]=pixes_copy[index];
		}
	}
}

__global__ void compactPos(int *idata,int *scanned, int *pos,int *pos_copy,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(idata[index]==1){
			pos[scanned[index]]=pos_copy[index];
		}
	}
}

__global__ void multiplyColor(int *idata,glm::vec3 *pixes,glm::vec3 *images,int *pos,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		if(idata[index]==0){
			images[pos[index]]=pixes[index];
		}
	}
}

int compact(int n, glm::vec3 *origin, glm::vec3 *direction, int *pos,glm::vec3 *pixes,
			glm::vec3 *image,bool *inside,int *idata,int blockSize) {
    int *scanned=new int[n];
	scan(n,scanned,idata,blockSize);
	int *dev_idata,*dev_scanned;
	glm::vec3 *origin_copy,*direction_copy,*pixes_copy;
	int *pos_copy;
	bool *inside_copy;

	cudaMalloc(&dev_idata,n*sizeof(int));
	cudaMalloc(&dev_scanned,n*sizeof(int));
	cudaMemcpy(dev_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_scanned,scanned,n*sizeof(int),cudaMemcpyHostToDevice);

	cudaMalloc(&origin_copy,n*sizeof(glm::vec3));
	cudaMalloc(&direction_copy,n*sizeof(glm::vec3));
	cudaMalloc(&pixes_copy,n*sizeof(glm::vec3));
	cudaMalloc(&pos_copy,n*sizeof(int));
	cudaMalloc(&inside_copy,n*sizeof(bool));
	cudaMemcpy(origin_copy,origin,n*sizeof(glm::vec3),cudaMemcpyDeviceToDevice);
	cudaMemcpy(direction_copy,direction,n*sizeof(glm::vec3),cudaMemcpyDeviceToDevice);
	cudaMemcpy(pixes_copy,pixes,n*sizeof(glm::vec3),cudaMemcpyDeviceToDevice);
	cudaMemcpy(pos_copy,pos,n*sizeof(int),cudaMemcpyDeviceToDevice);
	cudaMemcpy(inside_copy,inside,n*sizeof(bool),cudaMemcpyDeviceToDevice);

	dim3 gridSize((n+blockSize-1)/blockSize);
	multiplyColor<<<gridSize,blockSize>>>(dev_idata,pixes,image,pos,n);
	compactOrigin<<<gridSize,blockSize>>>(dev_idata,dev_scanned,origin,origin_copy,n);
	compactDirection<<<gridSize,blockSize>>>(dev_idata,dev_scanned,direction,direction_copy,n);
	compactPos<<<gridSize,blockSize>>>(dev_idata,dev_scanned,pos,pos_copy,n);
	compactInside<<<gridSize,blockSize>>>(dev_idata,dev_scanned,inside,inside_copy,n);
	compactPixes<<<gridSize,blockSize>>>(dev_idata,dev_scanned,pixes,pixes_copy,n);

	cudaFree(dev_idata);
	cudaFree(dev_scanned);
	cudaFree(origin_copy);
	cudaFree(direction_copy);
	cudaFree(pos_copy);
	cudaFree(pixes_copy);
	cudaFree(inside_copy);

	int count=scanned[n-1]+idata[n-1];
	delete(scanned);

	return count;
}