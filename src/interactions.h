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
		bool out,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	
	
	if (m.hasRefractive == 1) {
		thrust::uniform_real_distribution<float> probDistrib(0.0f, 1.0f);
		float prob = probDistrib(rng);
		float angle;
		glm::vec3 refractionPoint;
		
		
		float R_0;
		if (out) {
			R_0 = ((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction))*((1.0f - m.indexOfRefraction) / (1.0f + m.indexOfRefraction));
			float n_12 = glm::pow(1.0f / m.indexOfRefraction, 2);
			angle = 1.0f - n_12 * (1.0f - glm::pow(glm::dot(normal, ray.direction), 2));
		}
		else {
			R_0 = ((m.indexOfRefraction - 1.0f) / (m.indexOfRefraction + 1.0f)) * ((m.indexOfRefraction - 1.0f) / (m.indexOfRefraction + 1.0f));
			angle = 1.0f - glm::pow(m.indexOfRefraction, 2) * (1.0f - glm::pow(glm::dot(normal, ray.direction), 2));
		}
		float reflCoeff = R_0 + (1.0f - R_0) * glm::pow(1.0f - glm::dot(normal, -ray.direction), 5); 
		if ((1.0f - reflCoeff) > prob && angle > 0.0f) {
			if (out == true){ //if ray is coming from air to geom
				refractionPoint = glm::refract(ray.direction, normal, 1.0f / m.indexOfRefraction);
				ray.out = false;
			}
			else{ //if ray is coming out of geom into air
				refractionPoint = glm::refract(ray.direction, normal, m.indexOfRefraction);
				ray.out = true;
			}
			ray.direction = refractionPoint;
			ray.origin = intersect + glm::vec3(0.01f, 0.01f, 0.01f)*(glm::normalize(ray.direction));
			color *= m.color;// *(1.0f / reflCoeff);
			
		}
		else {
			ray.direction = ray.direction - 2.0f*normal*(glm::dot(ray.direction, normal));
			ray.origin = intersect + glm::vec3(0.01f, 0.01f, 0.01f)*(glm::normalize(ray.direction));
			color *= m.color;// *(1.0f / reflCoeff);
		}

	}
	else if (m.hasReflective == 1) {
		ray.direction = ray.direction - 2.0f*normal*(glm::dot(ray.direction, normal));
		ray.origin = intersect + glm::vec3(0.01f, 0.01f, 0.01f)*(glm::normalize(ray.direction));
		color *= m.color;
	}
	else if (m.specular.exponent > 0) {
		thrust::uniform_real_distribution<float> probDistrib(0.0f, 1.0f);
		float prob = probDistrib(rng);
		if (.1f > prob) {
			ray.direction = ray.direction - 2.0f*normal*(glm::dot(ray.direction, normal));
			ray.origin = intersect + glm::vec3(0.01f, 0.01f, 0.01f)*(glm::normalize(ray.direction));
			color *= m.specular.color;
		}
		else {
			ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			ray.origin = intersect + glm::vec3(0.01f, 0.01f, 0.01f)*(glm::normalize(ray.direction));
			color *= m.color;
		}
	}
	//DIFFUSE
	else {
		ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		ray.origin = intersect + glm::vec3(0.01f, 0.01f, 0.01f)*(glm::normalize(ray.direction));
		color *= m.color;
	}

}
