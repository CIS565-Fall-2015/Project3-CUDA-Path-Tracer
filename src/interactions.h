#pragma once

#include "intersections.h"
#include "sceneStructs.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float SchlickCosine(float cosTheta, float IORi, float IORt) {
	float R0 = ((IORi - IORt) / (IORi + IORt));
	R0 *= R0;
	return R0 + (1.0f - R0) * pow((1.0f - cosTheta), 5.0f);
}

__host__ __device__ void reflect(
	PathRay &rayStep,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	float term
	) {
	// perfect reflection: http://paulbourke.net/geometry/reflected/
	rayStep.ray.direction = rayStep.ray.direction - 2.0f * normal *
		(glm::dot(rayStep.ray.direction, normal));
	rayStep.ray.origin = intersect;
	//if (m.hasRefractive) {
	//	// compute using Schlick's approximation
	//	float cosTheta = glm::dot(rayStep.ray.direction, normal);
	//	if (cosTheta > 0.0f) {
	//		float fresnel = SchlickCosine(cosTheta, 1.0f, m.indexOfRefraction);
	//		glm::vec3 color = m.color;
	//		color *= term * fresnel;
	//		rayStep.color *= color;
	//		return;
	//	}
	//}
	glm::vec3 color = m.color;
	color *= term;
	rayStep.color *= color;
}

__host__ __device__ void refract(
	PathRay &rayStep,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material &m,
	float term
	) {
	// refract: http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
	float IORi = 1.0f; // THE VOID
	float IORt = m.indexOfRefraction;
	float cosAnglei = glm::dot(rayStep.ray.direction, normal);

	float sinAnglei = sqrt(1.0f - cosAnglei * cosAnglei);
	if (sinAnglei > IORt / IORi) { // total internal reflection
		rayStep.depth = 0;
		rayStep.color[0] = 0;
		rayStep.color[1] = 0;
		rayStep.color[2] = 0;
		return;
	}

	float underSqrt = 1.0f - (((IORi * IORi) / (IORt * IORt)) * (1.0f - cosAnglei * cosAnglei));

	if (underSqrt > 0.0f) {
		rayStep.ray.direction = (IORi / IORt) * rayStep.ray.direction;
		rayStep.ray.direction += ((IORi / IORt) * cosAnglei - sqrt(underSqrt)) * normal;
	}
	else {
		rayStep.depth = 0;
		rayStep.color[0] = 0;
		rayStep.color[1] = 0;
		rayStep.color[2] = 0;
		return;
	}

	rayStep.ray.origin = intersect + 0.001f * rayStep.ray.direction;
	glm::vec3 color = m.color;
	color *= term;
	rayStep.color *= color;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways:
 * - Always take a 50/50 split between a diffuse bounce and a specular bounce,
 *   but multiply the result of either one by 1/0.5 to cancel the 0.5 chance
 *   of it happening.
 * - Pick the split based on the intensity of each color, and multiply each
 *   branch result by the inverse of that branch's probability (same as above).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathRay &rayStep,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	// premultiply color.
	if (m.emittance > 0.0f) { // hitting a light
		rayStep.color *= m.color * m.emittance;
		rayStep.depth = 0;
	}
	else if (rayStep.depth <= 0){ // bottoming out
		rayStep.color = glm::vec3(0, 0, 0);
	}
	//else if (m.hasReflective > 0.0f && m.hasRefractive > 0.0f) {
	//	thrust::uniform_real_distribution<float> u01(0, 1);
	//	// 50/50 split where the ray goes
	//	if (u01(rng) <= 0.5f) {
	//		refract(rayStep, intersect, normal, m, 1.0f);
	//	}
	//	else {
	//		reflect(rayStep, intersect, normal, m, 1.0f);
	//	}
	//
	//}
	else if (m.hasReflective > 0.0f && m.hasRefractive < 1.0f) { // hitting a simple mirrored object
		reflect(rayStep, intersect, normal, m, 1.0f);
	}
	else if (m.hasRefractive > 0.0f && m.hasReflective < 1.0f) { // hitting a simple refractive object
		refract(rayStep, intersect, normal, m, 1.0f);
	}
	// basic diffuse. "deploy a new ray" in a random cosine weighted direction.
	else // hitting just a normal thing
	{
		rayStep.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		rayStep.ray.origin = intersect;
		rayStep.color *= m.color;
	}
}