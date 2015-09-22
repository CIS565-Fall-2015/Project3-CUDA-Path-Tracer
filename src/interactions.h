#pragma once

#include "intersections.h"

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

__host__ __device__
glm::vec3 calculateRandomSpecDirection(
glm::vec3 normal, float n, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = pow(u01(rng), 1/(n+1)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;	// phi

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
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

__host__ __device__
glm::vec3 calculatePerfectSpecDirection(glm::vec3 inDirection, glm::vec3 normal) {
	return inDirection - 2.0f * glm::dot(inDirection, normal) * normal;
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
        Ray &ray,
        glm::vec3 &color,
		glm::vec3 intersect,
		glm::vec3 inDirection,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	float atten;

	glm::vec3 diffuseDirection = calculateRandomDirectionInHemisphere(normal, rng);
	atten = dot(normalize(diffuseDirection), normalize(normal));
	if (atten < 0){
		atten = 0;
	}
	glm::vec3 diffuseColor = color * m.color * atten;

	float split = 0.5;

	glm::vec3 perfectSpecDirection = calculatePerfectSpecDirection(inDirection, normal);
	atten = dot(normalize(perfectSpecDirection), normalize(normal));
	if (atten < 0){
		atten = 0;
	}
	glm::vec3 specColor = color * m.specular.color * atten;

	glm::vec3 glossNormal = calculateRandomSpecDirection(normal, m.specular.exponent, rng);
	glm::vec3 glossDirection = calculatePerfectSpecDirection(inDirection, glossNormal);
	atten = dot(normalize(glossDirection), normalize(normal));
	if (atten < 0){
		atten = 0;
	}
	glm::vec3 glossColor = color * m.specular.color * atten;


	thrust::uniform_real_distribution<float> range(0, 1);
	float pick = range(rng);

	if (pick < split){
		color = diffuseColor * 2.0f;
		ray.direction = diffuseDirection;
	}
	else {
		if (m.hasReflective == 1.0f){
			color = specColor * 2.0f;
			ray.direction = perfectSpecDirection;
		}
		else {
			color = glossColor * 2.0f;
			ray.direction = glossDirection;
		}
	}

	ray.origin = intersect;
}
