#pragma once

#include <glm/glm.hpp>
#include "glm/gtx/intersect.hpp"

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
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

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
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
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
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
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

__host__ __device__ float triangle_intersection(Ray &ray_world, glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3)
{

	glm::vec3 dir = ray_world.direction;
	glm::vec3 ori = ray_world.origin;



	glm::vec3 e1, e2;
	e1 = v2 - v1;
	e2 = v3 - v1;

	glm::vec3 P = glm::cross(dir, e2);
	float det = glm::dot(e1, P);

	float m_EPSILON = 1.e-8;
	if (det > -m_EPSILON && det < m_EPSILON)
	{
		return -1;
	}

	float inv_det = 1.f / det;

	glm::vec3 T = ori - v1;

	float u = glm::dot(T, P)*inv_det;

	if (u<0.f || u>1.f)
	{
		return -1;
	}

	glm::vec3 Q = glm::cross(T, e1);

	float v = glm::dot(dir, Q)*inv_det;

	if (v<0.f || u + v>1.f)
	{
		return -1;
	}

	float t = glm::dot(e2, Q)*inv_det;

	//return t;
	if (t > m_EPSILON)
	{
		return t;
	}


	return -1;




}



__host__ __device__ float OBJMeshIntersectionTest(cuda_OBJMesh_head mesh, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {

	//bounding box to pre check the intersection
	float cur_t = boxIntersectionTest(mesh.m_box, r, intersectionPoint, normal, outside);

	if (cur_t == -1) return -1;

	cur_t = -1;

	int triangle_index = -1;
	int num_of_triangles = mesh.num_of_tri;

	for (unsigned int i = 0; i<num_of_triangles; i++)
	{
		glm::vec3 v1 = mesh.m_v[mesh.m_tri_list[3 * i]];
		glm::vec3 v2 = mesh.m_v[mesh.m_tri_list[3 * i + 1]];
		glm::vec3 v3 = mesh.m_v[mesh.m_tri_list[3 * i + 2]];

		float tmp_t = triangle_intersection(r, v1, v2, v3);
		
		/*if (tmp_t == -1)
		tmp_t = triangle_intersection(r, v1, v3, v2);*/

		

		if (tmp_t!=-1)
		{
			if (cur_t == -1)
			{
				cur_t = tmp_t;
				triangle_index = i;
			}
			else if (tmp_t < cur_t)
			{
				cur_t = tmp_t;
				triangle_index = i;
			}
			
		}
		else
		{
			continue;
		}
	}

	if (cur_t == -1) 
	{
		return -1;
	}

	intersectionPoint = r.origin + r.direction * (cur_t - 0.00001f);
	
	
	glm::vec3 cur_normal = glm::normalize(mesh.m_n[triangle_index]);
	float dot_product = glm::dot(cur_normal, r.direction);
	normal = dot_product > 0 ? -cur_normal : cur_normal;
	outside = dot_product > 0 ? false : true;

	return cur_t;
}

