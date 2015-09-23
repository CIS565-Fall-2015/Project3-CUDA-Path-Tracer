#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3& intersectionPoint, glm::vec3& normal) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));
    bool outside;
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		//normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
        normal = glm::normalize(multiplyMV(box.transform,
                    glm::vec4(outside ? tmin_n : tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3& intersectionPoint, glm::vec3& normal) {
    bool outside = false;
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool cubeImpl(Ray r,float xmax,float xmin,float ymax,float ymin,float zmax,float zmin){
	glm::vec3 normal(0,0,0);
	glm::vec3 intPoint;
	float t=-1,tx1,tx2,ty1,ty2,tz1,tz2,t1,t2,temp;
	int lab1,lab2,lab;
	
	if(r.direction.x==0){
		float x=r.origin.x;
		tx1=-1000000;tx2=1000000;
		if(x>xmax||x<xmin) return false;
	}
	if(r.direction.y==0){
		float y=r.origin.y;
		ty1=-1000000;ty2=1000000;
		if(y>ymax||y<ymin) return false;
	}
	if(r.direction.z==0){
		float z=r.origin.z;
		tz1=-1000000;tz2=1000000;
		if(z>zmax||z<zmin) return false;
	}
	
	if(r.direction.x>0){
		tx1=(xmin-r.origin.x)/r.direction.x;
		tx2=(xmax-r.origin.x)/r.direction.x;
	}
	else if(r.direction.x<0){
		tx2=(xmin-r.origin.x)/r.direction.x;
		tx1=(xmax-r.origin.x)/r.direction.x;
	}
	//x
	if(r.direction.y>0){
		ty1=(ymin-r.origin.y)/r.direction.y;
		ty2=(ymax-r.origin.y)/r.direction.y;
	}
	else if(r.direction.y<0){
		ty2=(ymin-r.origin.y)/r.direction.y;
		ty1=(ymax-r.origin.y)/r.direction.y;
	}
	//y
	if(r.direction.z>0){
		tz1=(zmin-r.origin.z)/r.direction.z;
		tz2=(zmax-r.origin.z)/r.direction.z;
	}
	else if(r.direction.z<0){
		tz2=(zmin-r.origin.z)/r.direction.z;
		tz1=(zmax-r.origin.z)/r.direction.z;
	}
	//z
	if(tx1>ty1){
		t1=tx1;
		lab1=1;
	}
	else{
		t1=ty1;
		lab1=2;
	}
	if(t1<tz1){
		t1=tz1;
		lab1=3;
	}
	//t1
	if(tx2<ty2){
		t2=tx2;
		lab2=1;
	}
	else{
		t2=ty2;
		lab2=2;
	}
	if(t2>tz2){
		t2=tz2;
		lab2=3;
	}
	//t2;
	if(t1>t2) return false;//no intersect
	if(t2<0) return false;//negative intersect*/
	return true;
}

/*__host__ __device__ void kdIntersect(Ray ray,kdtree *root){
	//if(root==nullptr||!cubeImpl(ray,root->xmax,root->xmin,root->ymax,root->ymin,root->zmax,root->zmin)) return;//not intersect
	//int a=root->ymax+root->xmax+root->zmax+root->xmin+root->ymin+root->zmin;
	//if(count>=50) return;
	//if(root->lc==nullptr&&root->rc==nullptr) treeIdx[count++]=root->mesh;
	//else{
		if(root->lc!=nullptr) kdIntersect(ray,root->lc);
		if(root->rc!=nullptr) kdIntersect(ray,root->rc);
	//}
}*/

__host__ __device__ void kdIntersect(Ray r, kdtree *root,int *treeIdx){
	int count=0,num=0,n=0;
	kdtree *kd[500];
	kd[count++]=root;
	while(count<500&&n!=count){
		kdtree *current=kd[n];
		if(cubeImpl(r,current->xmax,current->xmin,current->ymax,current->ymin,current->zmax,current->zmin)){
			if(current->lc==nullptr&&current->rc==nullptr) treeIdx[num++]=current->mesh;
			else{
				kd[count++]=current->lc;
				if(count>=500) break;
				kd[count++]=current->rc;
			}
		}
		n++;
	}
}

__host__ __device__ float triIntersectionTest(Ray r, glm::vec3 p1,glm::vec3 p2,glm::vec3 p3,glm::vec3& normal){
	float t=-1;
	glm::vec3 v1,v2,v;
	v1=p2-p1;v2=p3-p2;
	v=glm::cross(v1,v2);
	if(v.length()<1e-15) return -1;
	float d=-(r.origin.x-p2.x)*v.x-(r.origin.y-p2.y)*v.y-(r.origin.z-p2.z)*v.z;
	float s=v.x*r.direction.x+v.y*r.direction.y+v.z*r.direction.z;
	if(fabs(s)<1e-15) return -1;//not intersect
	t=d/s;
	
	if(t<0) return -1;//negative intersect
	float r1,r2,r3;
	glm::vec3 center=r.origin+t*r.direction;
	glm::vec3 c1,c2,c3;
	c1=p1-center;c2=p2-center;c3=p3-center;
	r1=(glm::dot(c1,c1)+glm::dot(c2,c2)-glm::dot(p2-p1,p2-p1))/(2*glm::length(c1)*glm::length(c2));
	r2=(glm::dot(c1,c1)+glm::dot(c3,c3)-glm::dot(p3-p1,p3-p1))/(2*glm::length(c1)*glm::length(c3));
	r3=(glm::dot(c2,c2)+glm::dot(c3,c3)-glm::dot(p2-p3,p2-p3))/(2*glm::length(c2)*glm::length(c3));
	glm::vec3 edge1=p2-p1;
	glm::vec3 edge2=p3-p1;
	glm::vec3 p=r.origin+t*r.direction;
	float e11=glm::dot(edge1,edge1);
	float e12=glm::dot(edge1,edge2);
	float e22=glm::dot(edge2,edge2);
	glm::vec3 w=p-p1;
	float we1=glm::dot(w,edge1);
	float we2=glm::dot(w,edge2);

	float D=e12*e12-e11*e22;
	s=(e12*we2-e22*we1)/D;
	float t0=(e12*we1-e11*we2)/D;
	if(s<0||s>1||t0<0||(s+t0)>1){
		return -1;
	}
	normal=glm::normalize(v);
	return t;
}

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal
				) {
	float t=-1;
	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
	Ray rt;
    rt.origin = ro;
    rt.direction = rd;
	int treeIdx[100];
	for(int i=0;i<100;++i) treeIdx[i]=-1;
	kdIntersect(rt,mesh.mesh->tree,treeIdx);
	for(int i=0;i<100;++i){
		if(treeIdx[i]==-1) break;
		//int index1=mesh.mesh->indices[i];
		//int index2=mesh.mesh->indices[i+1];
		//int index3=mesh.mesh->indices[i+2];
		glm::vec3 p1=mesh.mesh->vertex[mesh.mesh->indices[3*treeIdx[i]]];
		glm::vec3 p2=mesh.mesh->vertex[mesh.mesh->indices[3*treeIdx[i]+1]];
		glm::vec3 p3=mesh.mesh->vertex[mesh.mesh->indices[3*treeIdx[i]+2]];
		glm::vec3 tmpNormal;
		float tmp=triIntersectionTest(rt,p1,p2,p3,tmpNormal);
		//float tmp=triIntersectionTest(rt,mesh.mesh->vertex[index1],mesh.mesh->vertex[index2],mesh.mesh->vertex[index3],tmpNormal);
		if(tmp!=-1&&(t==-1||tmp<t)){
			normal=tmpNormal;
			t=tmp;
		}
	}
	if(t==-1) return t;
	glm::vec3 objspaceIntersection=getPointOnRay(rt,t);

	intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
	normal = glm::normalize(multiplyMV(mesh.transform, glm::vec4(normal,0.0f)));
	return glm::length(r.origin - intersectionPoint);
}
