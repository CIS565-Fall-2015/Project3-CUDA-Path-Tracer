#pragma once

#include "intersections.h"

#define  FRESNEL_ENABLE

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
glm::vec3 calculateSpecularDirection( float n, glm::vec3 normal, thrust::default_random_engine &rng) 
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float sigma1 = u01(rng);
	float sigma2 = u01(rng);

	float cos_theta = pow((double)sigma1, (double)1.0 / (double)(n + 1));
	float sin_theta = sqrt(1 - cos_theta * cos_theta);
	float phi = sigma2 * TWO_PI;

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


	return cos_theta * normal
		+ cos(phi) * sin_theta * perpendicularDirection1
		+ sin(phi) * sin_theta * perpendicularDirection2;

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
		bool outside,
		bool &is_total_internalReflection,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	if (m.hasReflective) //reflect
	{
		
		glm::vec3 dir_in = ray.direction;
		glm::vec3 dir_out = (-glm::dot(dir_in,normal))*normal*2.f + dir_in;
		
		ray.direction = calculateSpecularDirection(m.specular.exponent, glm::normalize(dir_out), rng);


	}
	else if (m.hasRefractive) //refract 
	{
		glm::vec3 dir_in = ray.direction;
		float cos_theta1 = -glm::dot(dir_in, normal);
		float sin_theta1 = sqrt(1 - cos_theta1*cos_theta1);

		float sin_theta2;
		float cos_theta2;
		//snell's law
		if (outside) //from air to material n1 =1 n2 = m.indexOfRefraction
		{
			sin_theta2 = sin_theta1 / m.indexOfRefraction;
			cos_theta2 = sqrt(1 - sin_theta2*sin_theta2);

#ifdef FRESNEL_ENABLE
			float R1 = (cos_theta1 - m.indexOfRefraction*cos_theta2) / (cos_theta1 + m.indexOfRefraction*cos_theta2);
			R1*=R1;

			float R2 = (m.indexOfRefraction*cos_theta1 - cos_theta2) / (m.indexOfRefraction*cos_theta1 + cos_theta2);
			R2 *= R2;

			float R = (R1 + R2) / 2;
			thrust::uniform_real_distribution<float> u01(0, 1);
			float rng_num = u01(rng);

			if (rng_num <= R)
			{
				glm::vec3 dir_out = (-glm::dot(dir_in, normal))*normal*2.f + dir_in;

				ray.direction = dir_out;
				is_total_internalReflection = true; //using this tag to diff when update the ray origin
			}
			else
			{
				ray.direction = cos_theta2 * (-normal) + (dir_in - cos_theta1*(-normal)) / sin_theta1*sin_theta2;
			}
			



#else
			ray.direction = cos_theta2 * (-normal) + (dir_in - cos_theta1*(-normal)) / sin_theta1*sin_theta2;

#endif // FRESNEL_ENABLE

			}
		else // form material to air (need to take care of the total internal reflection)
		{
			sin_theta2 = sin_theta1 * m.indexOfRefraction;
			if (sin_theta2 >= 1.f) //total internal reflection
			{
				glm::vec3 dir_out = (-glm::dot(dir_in, normal))*normal*2.f + dir_in;

				ray.direction =dir_out;
				is_total_internalReflection = true;

			}
			else
			{
				cos_theta2 = sqrt(1 - sin_theta2*sin_theta2);

#ifdef FRESNEL_ENABLE
				float R1 = (cos_theta1 - m.indexOfRefraction*cos_theta2) / (cos_theta1 + m.indexOfRefraction*cos_theta2);
				R1 *= R1;

				float R2 = (m.indexOfRefraction*cos_theta1 - cos_theta2) / (m.indexOfRefraction*cos_theta1 + cos_theta2);
				R2 *= R2;

				float R = (R1 + R2) / 2;
				thrust::uniform_real_distribution<float> u01(0, 1);
				float rng_num = u01(rng);

				if (rng_num <= R)
				{
					glm::vec3 dir_out = (-glm::dot(dir_in, normal))*normal*2.f + dir_in;

					ray.direction = dir_out;
					is_total_internalReflection = true; ////using this tag to diff when update the ray origin
				}
				else
				{
					ray.direction = cos_theta2 * (-normal) + (dir_in - cos_theta1*(-normal)) / sin_theta1*sin_theta2;
				}
#else
				ray.direction = cos_theta2 * (-normal) + (dir_in - cos_theta1*(-normal)) / sin_theta1*sin_theta2;

#endif // FRESNEL_ENABLE
			}
		}
		
	}
	else //diffuse
	{
		ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		ray.direction = glm::normalize(ray.direction);
	}
}
