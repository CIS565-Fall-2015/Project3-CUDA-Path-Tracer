#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        const glm::vec3 normal, thrust::default_random_engine &rng) {
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

__device__ void refract(glm::vec3 incident, glm::vec3 normal, float n1, float n2){

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
__device__
void scatterRay(
        Ray &ray,
        const glm::vec3 intersect,
        const glm::vec3 normal,
        const Material &m,
		const Geom &g,
        thrust::default_random_engine &rng) {
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float randNum = u01(rng);

	// Light
	if (m.emittance > 0.0f){
		ray.color = ray.color * m.emittance;
		ray.isAlive = false;
		return;
	}

	// From Stanford notes
	if (m.hasRefractive){
		float n1 = 1.0f;
		float n2 = m.indexOfRefraction;

		float R0 = (n1 - n2) / (n1 + n2);
		R0 *= R0;
		float cosi = -glm::dot(normal, ray.direction);
		float cosii = 1.0f - cosi;
		float RT = R0 + (1.0f-R0)*cosii*cosii*cosii*cosii*cosii;

		// Specular
		if (randNum < RT){
			ray.direction = ray.direction - 2.0f * (glm::dot(ray.direction, normal)) * normal;
			ray.color = ray.color * m.specular.color * (1.0f/RT);
			ray.origin = intersect;
		}
		else{
			float snell_ratio = n1 / n2;
			float cosT = sqrt(1.0f - snell_ratio*snell_ratio*(1.0f - cosi*cosi));
			ray.direction = snell_ratio*ray.direction + (snell_ratio * cosi - cosT)*normal;
			//ray.direction = glm::refract(ray.direction, normal, snell_ratio);

			//ray.color = ray.color * (1.0f - RT);
			//ray.color = ray.color * m.specular.color * (1.0f/(1.0f - RT));
			
			//ray.origin = intersect;//+0.0002f*ray.direction;
			ray.origin = intersect + 0.001f*ray.direction;
			glm::vec3 new_intersect;
			glm::vec3 new_normal;
			bool outside;
			// TODO: Intersect it again
			if (g.type == SPHERE){
				sphereIntersectionTest(g, ray, new_intersect, new_normal, outside);
			}
			else {
				boxIntersectionTest(g, ray, new_intersect, new_normal, outside);
			}
			ray.direction = glm::refract(ray.direction, new_normal, n2/n1);
			ray.origin = new_intersect + 0.001f*ray.direction;
			//ray.origin = new_intersect;// +0.0002f*ray.direction;
			//printf("%f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z);
		}
		
		//ray.origin = intersect;
		return;
	}

	float specProb = m.specular.color.r + m.specular.color.g + m.specular.color.b;
	float diffuseProb = m.color.r + m.color.g + m.color.b;
	float total = specProb + diffuseProb;
	specProb = specProb / total;

	//if (specProb < 0.00001f){
	//	specProb = 0.00001f;
	//}

	// Specular
	if (randNum < specProb){
		ray.direction = ray.direction - 2.0f * (glm::dot(ray.direction, normal)) * normal;
		ray.color = ray.color * m.specular.color * (1.0f/specProb);
	}
	// Diffuse
	else{
		ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		ray.color = ray.color * m.color * (1.0f/(1.0f-specProb));
	}
	ray.origin = intersect;
}