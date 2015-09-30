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

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * - (NOT RECOMMENDED - converges slowly or badly especially for pure-diffuse
 *   or pure-specular. In principle this correct, though.)
 *   Always take a 50/50 split between a diffuse bounce and a specular bounce,
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
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float randNum = u01(rng);

	float specularProb = m.specular.color.r + m.specular.color.g + m.specular.color.b;
	float diffuseProb = m.color.r + m.color.g + m.color.b;
	float total = specularProb + diffuseProb;
	specularProb = specularProb / total;
	diffuseProb = diffuseProb/total;
	
	// Reflective
	if (m.hasReflective){

		ray.origin = intersect + normal * FLT_EPSILON;
		ray.direction = glm::reflect(ray.direction, normal);
		ray.isOutside = true;
		color *= m.color;

	} else if(m.hasRefractive) {
	// Refractive
		float r0, c, rTheta, tTheta;
		float ior = m.indexOfRefraction;

		if (ray.isOutside) {
			ior = 1.0f/ior;
			r0 = glm::pow((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction), 2);
			c = 1.0f - glm::dot(-ray.direction, normal);
			rTheta = r0 + (1.0f - r0) * glm::pow(c, 5);
			tTheta = 1.0f - rTheta;
		} else {
			r0 = glm::pow((m.indexOfRefraction - 1.0f) / (1.0f + m.indexOfRefraction), 2);
			c = 1.0f - glm::dot(-ray.direction, normal);
			rTheta = r0 + (1.0f - r0) * glm::pow(c, 5);
			tTheta = 1.0f - rTheta;
		}

		ray.isOutside = !ray.isOutside;
		
		float randomNum = u01(rng);
		if (randomNum <= rTheta) {
			ray.direction = glm::reflect(calculateRandomDirectionInHemisphere(normal, rng), 
				glm::normalize(normal));
			color *= m.color;
			
		} else {
			ray.direction = glm::refract(ray.direction, glm::normalize(normal), ior);
			color *= m.color;
		}
		
		ray.origin = intersect + normal + ray.direction * 0.001f;


	} else {
	// Diffuse
		ray.origin = intersect;
		if(m.specular.exponent > 0 && specularProb < randNum) {
			glm::vec3 h = glm::vec3(0.0f, 1.0f, 0.0f);
			ray.direction = glm::reflect(h, normal);
			float specTerm = glm::pow(glm::dot(normal, h), m.specular.exponent);
			color *= m.specular.color * specTerm;
		} else {
			ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			color *= m.color;
		}
	}

	if (m.emittance > 0) {
		color *= m.emittance;
	}

	
}
