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
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
glm::vec3 scatterRay(
        Ray &ray,
        float intrT,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rrr) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

		if (m.emittance>0)
		{
			ray.carry *= m.emittance*m.color;//???? is this right....?
			ray.terminated = true;
		}
		// Shading 
		else if (m.hasReflective || m.hasRefractive)
		{
			//!!! later : reflective or refractive
			if (m.hasReflective)
			{
				ray.origin = getPointOnRay(ray, intrT);
				ray.direction = glm::reflect(ray.direction, normal);
				ray.carry *= m.specular.color;
			}
			if (m.hasRefractive)
			{
				//rays[index].origin = getPointOnRay(rays[index], intrT);
				//rays[index].direction = glm::reflect(rays[index].direction, intrNormal);
				//rays[index].carry *= intrMat.specular.color;
			}
			//later fresnel

		}
		else // diffuse / specular
		{
			//!!! later : specular
			//http://www.tomdalling.com/blog/modern-opengl/07-more-lighting-ambient-specular-attenuation-gamma/
		
			glm::vec3 newDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rrr));
			float specCoeff = 0;
			if (m.specular.exponent > 0)
			{
				glm::vec3 refl = glm::reflect(-newDir, normal);
				float cosAngle = max(0.0f, glm::dot(-ray.direction, refl));
				specCoeff = pow(cosAngle, m.specular.exponent);
				//specComp = specCoeff*intrMat.specular.color;
			}
			ray.origin = getPointOnRay(ray, intrT);
			ray.carry *= (m.color*(1 - specCoeff) + m.specular.color*specCoeff);
			ray.direction = newDir;
		}
		return ray.carry;

}
