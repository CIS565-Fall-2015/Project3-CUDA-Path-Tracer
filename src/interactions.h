#pragma once

#include "intersections.h"

//Taken from CIS 560 code
__host__ __device__
glm::vec3 getRandomPointOnCubeLight(Geom &box, thrust::default_random_engine &rng)
{
	glm::vec3 dim = multiplyMV(box.transform, glm::vec4(1,1,1,1));

	float side1 = dim[0] * dim[1];		// x-y
	float side2 = dim[1] * dim[2];		// y-z
	float side3 = dim[0] * dim[2];		// x-z
	float totalArea = 2.0f * (side1 + side2 + side3);

	// pick random face weighted by surface area
	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::uniform_real_distribution<float> u02(-0.5, 0.5);
	float r = u01(rng);
	// pick 2 random components for the point in the range (-0.5, 0.5)
	float c1 = u02(rng);
	float c2 = u02(rng);

	glm::vec4 point;
	if (r < side1 / totalArea) {
		// x-y front
		point = glm::vec4(c1, c2, 0.5f, 1);
	} else if (r < (side1 * 2) / totalArea) {
		// x-y back
		point = glm::vec4(c1, c2, -0.5f, 1);
	} else if (r < (side1 * 2 + side2) / totalArea) {
		// y-z front
		point = glm::vec4(0.5f, c1, c2, 1);
	} else if (r < (side1 * 2 + side2 * 2) / totalArea) {
		// y-z back
		point = glm::vec4(-0.5f, c1, c2, 1);
	} else if (r < (side1 * 2 + side2 * 2 + side3) / totalArea) {
		// x-z front
		point = glm::vec4(c1, 0.5f, c2, 1);
	} else {
		// x-z back
		point = glm::vec4(c1, -0.5f, c2, 1);
	}

	return multiplyMV(box.transform, point);
}

//Taken from CIS 560 code
__host__ __device__
glm::vec3 getRandomPointOnSphereLight(Geom &sphere, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float u = u01(rng);
	float v = u01(rng);

	float theta = 2.0f * PI * u;
	float phi = acos(2.0f * v - 1.0f);

	glm::vec4 point;
	point[0] = sin(phi) * cos(theta);
	point[1] = sin(phi) * sin(theta);
	point[2] = cos(phi);
	point[3] = 1.0;

	return multiplyMV(sphere.transform, point);
}

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
 *The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * (NOT RECOMMENDED - converges slowly or badly especially for pure-diffuse
 *   or pure-specular. In principle this correct, though.)
 *   Always take a 50/50 split between a diffuse bounce and a specular bounce,
 * - Always take a 50/50 split between a diffuse bounce and a specular bounce,
 *   but multiply the result of either one by 1/0.5 to cancel the 0.5 chance
 *   of it happening.
 * - Pick the split based on the intensity of each color, and multiply each
 *   branch result by the inverse of that branch's probability (same as above).

 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        RayState &ray,
        glm::vec3 intersect,
        glm::vec3 normal,
        Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.


//	if(m.emittance > 0)
//	{
//		//Light source
//		ray.isAlive = false;
//		ray.rayColor *= m.color;
//	}

	if(m.hasReflective == 0 && m.hasRefractive == 0)
	{
		//Diffused material
		ray.rayColor *= m.color;
		ray.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		ray.ray.origin = intersect + 0.0001f * ray.ray.direction;
	}
}
