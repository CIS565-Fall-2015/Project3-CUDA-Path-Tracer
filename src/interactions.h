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

//compute specular ray

__host__ __device__
glm::vec3 calculateRandomDirectionSpecular(
glm::vec3 normal, thrust::default_random_engine &rng, float shininess) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float theta = acos(powf(u01(rng), 1 / (shininess + 1)));
	float phi = TWO_PI * u01(rng);


	float x = cos(phi) * sin(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(theta);

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

	return z * normal
		+ y * perpendicularDirection1
		+ x * perpendicularDirection2;
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
bool scatterRay(
        Ray &ray,
        glm::vec3 &color,
        glm::vec3 intersect,
        glm::vec3 normal,
		bool outside,
        const Material &m, 
		thrust::default_random_engine rng,
		glm::vec2 uv,
		Texture* d_texture) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	glm::vec3 diffuseColor = m.color;
	if (m.textureid != -1){
		Texture t = d_texture[m.textureid];
		int x = uv[0] * t.width;
		int y = uv[1] * t.height;

		diffuseColor = t.d_img[x + y * t.width];
	}


	if (m.emittance > 0) {	//light source
		color *= m.color * m.emittance;
		return false;
	}

	if (m.hasRefractive > 0){
		glm::vec3 reflectDir = glm::reflect(ray.direction, normal);

		float r0 = pow((1 - m.indexOfRefraction) / (1 + m.indexOfRefraction), 2);
		float schlick = r0 + (1 - r0)*pow(1 - dot(normal, reflectDir), 5);

		thrust::uniform_real_distribution<float> u01(0, 1);
		float result = u01(rng);

		if (result < schlick){
			ray.direction = reflectDir;
			ray.origin = intersect + ray.direction * 0.001f;
			color *= m.specular.color;
		}
		else{
			float eta = m.indexOfRefraction;
			if (outside){
				eta = 1.0 / m.indexOfRefraction;
			}

			ray.direction = glm::refract(ray.direction, normal, eta);
			ray.origin = intersect + ray.direction * 0.1f;
		}
	}
	else if (m.hasReflective > 0){
		ray.direction = glm::reflect(ray.direction, normal);
		ray.origin = intersect + ray.direction * 0.001f;
		color *= diffuseColor;
	}
	else {
		//diffuse-specular
		thrust::uniform_real_distribution<float> u01(0, 1);
		float result = u01(rng);

		if (m.specular.exponent == 0 || result < 0.5){
			//diffuse
			ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
			ray.origin = intersect + ray.direction * 0.001f;
			color *= diffuseColor;
		}
		else{
			//specular
			ray.direction = calculateRandomDirectionSpecular(normal, rng, m.specular.exponent);
			ray.origin = intersect + ray.direction * 0.001f;
			color *= m.specular.color;
		}
	}
	return true;
}
