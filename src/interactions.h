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
* Computes Fresnel reflection coefficient using Schlick's approximation
* https://en.wikipedia.org/wiki/Schlick%27s_approximation
* http://www.scratchapixel.com/old/lessons/3d-basic-lessons/lesson-14-interaction-light-matter/optics-reflection-and-refraction/
*/
__host__ __device__
float calculateRefractionCoefficient(glm::vec3 direction, glm::vec3 normal, float indexOfRefraction) {
	//TODO: GLM pow instead?
	float r0 = glm::pow((1.0f - indexOfRefraction) / (1.0f + indexOfRefraction), 2);
	return r0 + (1.0f - r0) * glm::pow(1.0f - glm::dot(normal, -direction), 5);
}

/**
* Computes a reflection vector off a specular or refractive surface.
* Used for specular and refracted lighting.
*/
__host__ __device__
glm::vec3 calculateReflectionDirection(glm::vec3 direction, glm::vec3 normal) {
	return direction + 2.0f * glm::dot(-direction, normal) * normal;
}

/**
* Computes a refraction vector off a refractive surface.
* Used for refracted lighting.
*/
__host__ __device__
glm::vec3 calculateRefractionDirection(glm::vec3 direction, glm::vec3 normal, float angle, float eta) {
	return (-eta * glm::dot(normal, direction) - glm::sqrt(angle)) * normal + direction * eta;
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
	glm::vec3 diffuseColor = m.color;
	glm::vec3 specularColor = m.specular.color;
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (m.hasRefractive) {
		float reflectionCoefficient = calculateRefractionCoefficient(ray.direction, normal, m.indexOfRefraction);
		float refractionCoefficient = 1.0f - reflectionCoefficient;

		if (u01(rng) < refractionCoefficient) {
			float eta = 0.0f;
			if (!ray.inside) {
				eta = 1.0f / m.indexOfRefraction; // Coming from the air
			}
			float angle = 1.0f - glm::pow(eta, 2) * (1.0f - glm::pow(glm::dot(normal, ray.direction), 2));
			if (angle < 0.0f) {
				// Angle less than zero, so we reflect
				ray.direction = calculateReflectionDirection(ray.direction, normal);
				ray.origin = intersect + ray.direction * EPSILON;
				ray.inside = false;
			}
			else {
				// Here we do a refraction
				ray.direction = calculateRefractionDirection(ray.direction, normal, angle, eta);
				ray.origin = intersect + ray.direction * 0.001f; // For some reason epsilon is too small and gives bad results. Need to use larger one
				ray.inside = true;
			}
		}
		else {
			ray.direction = calculateReflectionDirection(ray.direction, normal);
			ray.origin = intersect + ray.direction * EPSILON;
			ray.inside = false;
		}
	}
	else if (m.hasReflective) {
		//First must determine if this is perfectly specular or not
		// is this only when there's an exponent? or when the diffuse is zero?
		// for now i will go with the exponent being non zero
		float specularExponent = m.specular.exponent;
		if (specularExponent != 0) {
			// non perfect
			/*
			 * This implementation is not working
			float thetaS, phiS;
			thrust::uniform_real_distribution<float> u01(0, 1);
			float xi1 = u01(rng), xi2 = u01(rng); //random values between 0 and 1
			glm::vec3 direction;
			
			thetaS = glm::acos(1.0f / (pow(xi1, specularExponent + 1)));
			phiS = 2.0f * PI * xi2;
			direction.x = glm::cos(phiS) * glm::sin(thetaS);
			direction.y = glm::sin(phiS) * glm::sin(thetaS);
			direction.z = glm::cos(thetaS);

			// this direction isn't in the correct coordinate system..
			// need to go from tangent to world and specular to gangent?
			ray.origin = intersect + normal * EPSILON;
			ray.direction = glm::normalize(direction);
			*/
			
			// Calculate intensity values
			float specularIntensity = (specularColor.x + specularColor.y + specularColor.z) / 3.0f;
			float diffuseIntensity = (diffuseColor.x + diffuseColor.y + diffuseColor.z) / 3.0f;

			float specularProbability = specularIntensity / (diffuseIntensity + specularIntensity);
			float diffuseProbability = diffuseIntensity / (diffuseIntensity + specularIntensity);

			float randColor = u01(rng);
			if (randColor <= specularProbability) {
				//spec
				ray.origin = intersect + normal * EPSILON;
				ray.direction = calculateReflectionDirection(ray.direction, normal);
				color *= specularColor * (1.0f / specularProbability);
				ray.inside = false;
			}
			else {
				//diffuse
				ray.origin = intersect + normal * EPSILON;
				ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
				color *= diffuseColor * (1.0f / diffuseProbability);
				ray.inside = false;
			}
		}
		else {
			// perfect mirror
			ray.origin = intersect + normal * EPSILON;
			ray.direction = calculateReflectionDirection(ray.direction, normal);
			color *= specularColor;
			ray.inside = false;
		}
	}
	else {
		// diffuse only
		ray.origin = intersect + normal * EPSILON;
		ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		color *= diffuseColor;
		ray.inside = false;
	}
}
