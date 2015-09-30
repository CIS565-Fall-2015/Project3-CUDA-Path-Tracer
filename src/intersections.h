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


__host__ __device__ float triangleIntersectionTest(Geom triangle, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside)
{
	glm::vec3 baryPosition;
	bool hit = glm::intersectRayTriangle<glm::vec3>(r.origin, r.direction
		, triangle.translation, triangle.rotation, triangle.scale, baryPosition);

	float t = baryPosition.z;

	//printf("%d  %f\n",(int)hit,t);

	if (t < 0)
	{
		return -1;
	}

	baryPosition.z = 1 - baryPosition.y - baryPosition.x;
	intersectionPoint = triangle.translation * baryPosition.x
		+ triangle.rotation * baryPosition.y
		+ triangle.scale * baryPosition.z;

	//normal vector?
	//use transform matrix?
	//interpolate the normal
	//[col][row]
	glm::vec3 n0(triangle.transform[0][0], triangle.transform[0][1], triangle.transform[0][2]);
	glm::vec3 n1(triangle.transform[1][0], triangle.transform[1][1], triangle.transform[1][2]);
	glm::vec3 n2(triangle.transform[2][0], triangle.transform[2][1], triangle.transform[2][2]);

	normal = glm::normalize(n0 * baryPosition.x + n1 * baryPosition.y + n2 * baryPosition.z);


	outside = glm::dot(-r.direction, normal) > 0;

	return t;
}



__host__ __device__ bool AABBIntersect(const AABB aabb,const Ray ray, float & tmin, float & tmax)
{
	//be sure the aabb is not empty
	glm::vec3 inv_dir(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
	
	float tx1 = (aabb.min_pos.x - ray.origin.x)*inv_dir.x;
	float tx2 = (aabb.max_pos.x - ray.origin.x)*inv_dir.x;

	//float tmin = min(tx1, tx2);
	//float tmax = max(tx1, tx2);
	tmin = min(tx1, tx2);
	tmax = max(tx1, tx2);

	float ty1 = (aabb.min_pos.y - ray.origin.y) * inv_dir.y;
	float ty2 = (aabb.max_pos.y - ray.origin.y) * inv_dir.y;

	tmin = max(tmin, min(ty1, ty2));
	tmax = min(tmax, max(ty1, ty2));


	if (tmax >= tmin)
	{
		float tz1 = (aabb.min_pos.z - ray.origin.z) * inv_dir.z;
		float tz2 = (aabb.max_pos.z - ray.origin.z) * inv_dir.z;

		tmin = max(tmin, min(tz1, tz2));
		tmax = min(tmax, max(tz1, tz2));


		//To confirm?? tmax>0
		if(tmax >= tmin && tmax > 0)
		{
			//the line of ray will intersect
			//but maybe the t < 0

			

			return true;
		}
	}


	return false;
}




