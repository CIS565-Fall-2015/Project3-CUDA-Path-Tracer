#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

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
    return r.origin + (t - .001f) * glm::normalize(r.direction); // using 0.0001f results in floating pt error!
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// recomputes transform, inverse transform, and transpose inverse transform
// for an object based on the given time in milliseconds and the object's velocity
__host__ __device__ void recomputeTransforms(Geom object, float t) {
	// check if there's a speed at all
	if (object.speed[0] + object.speed[1] + object.speed[2] < 0.001f) return;

	glm::vec3 translation_t = object.translation + object.speed * t;

	// recompute transform
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation_t);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), object.rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), object.rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), object.rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), object.scale);
	object.transform = translationMat * rotationMat * scaleMat;

	// recompute inverse
	// there's no reason this shouldn't work too :(
	//translationMat = glm::translate(glm::mat4(), -translation_t);
	//rotationMat = glm::rotate(glm::mat4(), -object.rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	//rotationMat = rotationMat * glm::rotate(glm::mat4(), -object.rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	//rotationMat = rotationMat * glm::rotate(glm::mat4(), -object.rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	//glm::vec3 inverseScale = glm::vec3(1.0f / object.scale.x, 1.0f / object.scale.y, 1.0f / object.scale.z);
	//scaleMat = glm::scale(glm::mat4(), inverseScale);
	//object.inverseTransform = scaleMat * rotationMat * translationMat;

	object.inverseTransform = glm::inverse(object.transform);

	// recompute inverse transpose
	object.invTranspose = glm::inverseTranspose(object.transform);
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
        glm::vec3& intersectionPoint, glm::vec3& normal, float time) {
	recomputeTransforms(box, time);

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
        normal = glm::normalize(multiplyMV(box.transform,
                    glm::vec4(outside ? tmin_n : -tmin_n, 0.0f)));
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
        glm::vec3& intersectionPoint, glm::vec3& normal, float time) {
	recomputeTransforms(sphere, time);

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

__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
		glm::vec3& intersectionPoint, glm::vec3& normal, float time) {
	recomputeTransforms(mesh, time);

	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	float nearestDistance = -1.0f;
	int triangleIndex = -1;
	for (int i = 0; i < mesh.numTriangles; i += 3) {
		glm::vec3 barycentric;
		bool intersects = false;
		intersects = glm::intersectRayTriangle(rt.origin, rt.direction, mesh.dev_triangleVertices[i],
			mesh.dev_triangleVertices[i + 1], mesh.dev_triangleVertices[i + 2], barycentric);
		// intersectRayTriangle gives z in barycentric as the distance along the ray.
		if (intersects && barycentric.z > 0.0f && 
			(barycentric.z < nearestDistance || nearestDistance < 0.0f)) {
			nearestDistance = barycentric.z;
			triangleIndex = i;
		}
	}
	if (triangleIndex > -1 && nearestDistance > 0.0f) {
		glm::vec3 local_normal;
		// compute normal using triangle's indices. assume cclockwise face.
		glm::vec3 sideA = mesh.dev_triangleVertices[triangleIndex + 1] - 
			mesh.dev_triangleVertices[triangleIndex];
		glm::vec3 sideB = mesh.dev_triangleVertices[triangleIndex + 2] -
			mesh.dev_triangleVertices[triangleIndex];
		sideA = glm::normalize(sideA);
		sideB = glm::normalize(sideB);
		local_normal = glm::cross(sideA, sideB);
		glm::vec3 objspaceIntersection = getPointOnRay(rt, nearestDistance);

		intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
		normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(local_normal, 0.f)));
		return glm::length(r.origin - intersectionPoint);

	}
	return -1;

};