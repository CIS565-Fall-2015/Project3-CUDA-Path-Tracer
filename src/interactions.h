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
float calculateFresnelReflectionCoefficient(Ray ray, glm::vec3 normal, float intersectIOR, float transmittedIOR) {
	float r0 = glm::pow((intersectIOR - transmittedIOR) / (intersectIOR + transmittedIOR), 2);

	// curious if this is between the direction of the ray or the intersection point, little confused
	// shouldn't need to modify the ray so not passing the address
	// starting to this this is wrong and should be the incident. will have to test
	return r0 + (1.0f - r0) * glm::pow(1 - glm::dot(normal, ray.direction), 5);
	//TRIPPLE CHECK THIS
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

	if (m.hasReflective && m.hasRefractive) {
		//both refractive and reflective

		// first need to calculate reflective coefficient using schlick;s
		// wait how do i know if i am in the air or not?
		float reflectionCoefficient, eta;
		if (glm::dot(ray.direction, normal) < 0.0f) {
			// The ray is outside the object
			reflectionCoefficient = calculateFresnelReflectionCoefficient(ray, normal, 1.0f, m.indexOfRefraction);
			eta = 1.0f / m.indexOfRefraction;
		}
		else {
			// The ray is inside the object
			// TODO/NOTE MAYBE THE NORMAL SHOULD BE INVERTED HERE?
			reflectionCoefficient = calculateFresnelReflectionCoefficient(ray, normal, m.indexOfRefraction, 1.0f);
			eta = m.indexOfRefraction / 1.0f; //this could be backwards for all i know
		}
		float refractionCoefficient = 1.0f - reflectionCoefficient;

		float rand = u01(rng);
		if (rand <= reflectionCoefficient) {
			// it reflects
			ray.origin = intersect + normal * EPSILON;
			ray.direction = ray.direction + 2.0f * glm::dot(-ray.direction, normal) * normal;
			color *= specularColor * (1.0f / refractionCoefficient);
		}
		else {
			// it refracts
			// origin or direction or a mix of both? no idea
			// night need to play with epsilon
			ray.origin = intersect + normal * EPSILON; // minus normal?
			ray.direction = glm::refract(ray.origin, normal, eta);
			color *= specularColor * (1.0f / reflectionCoefficient);
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

			ray.origin = intersect + normal * EPSILON;
			
			// Calculate intensity values
			float specularIntensity = (specularColor.x + specularColor.y + specularColor.z) / 3.0f;
			float diffuseIntensity = (diffuseColor.x + diffuseColor.y + diffuseColor.z) / 3.0f;

			float specularProbability = specularIntensity / (diffuseIntensity + specularIntensity);
			float diffuseProbability = diffuseIntensity / (diffuseIntensity + specularIntensity);

			float randColor = u01(rng);
			if (randColor <= specularProbability) {
				//spec
				ray.direction = ray.direction + 2.0f * glm::dot(-ray.direction, normal) * normal;
				color *= specularColor * (1.0f / specularProbability);
			}
			else {
				//diffuse
				ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
				color *= diffuseColor * (1.0f / diffuseProbability);
			}
		}
		else {
			// perfect mirror
			ray.origin = intersect + normal * EPSILON;
			ray.direction = ray.direction + 2.0f * glm::dot(-ray.direction, normal) * normal;
			color *= specularColor;
		}
	}
	else if (m.hasRefractive) {
		//refractive only
	}
	else {
		// diffuse only
		ray.origin = intersect + normal * EPSILON;
		ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		color *= diffuseColor;
	}

	// TODO: Add refraction. Can something be diffuse, refractive, and reflective? (glass). how to do...
}
