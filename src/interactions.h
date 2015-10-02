#pragma once

#include "intersections.h"
#include <glm/gtc/matrix_inverse.hpp>

//Taken from CIS 560 code
__host__ __device__
glm::vec3 getRandomPointOnCubeLight(Geom &box, thrust::default_random_engine &rng)
{
	glm::vec3 dim = box.scale;//, glm::vec4(1,1,1,1.0f));

	float side1 = dim[0] * dim[1];		// x-y
	float side2 = dim[1] * dim[2];		// y-z
	float side3 = dim[0] * dim[2];		// x-z
	float totalArea = 1.0f / (2.0f * (side1 + side2 + side3));

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
	} else if (r < (side1 * 2) * totalArea) {
		// x-y back
		point = glm::vec4(c1, c2, -0.5f, 1);
	} else if (r < (side1 * 2 + side2) * totalArea) {
		// y-z front
		point = glm::vec4(0.5f, c1, c2, 1);
	} else if (r < (side1 * 2 + side2 * 2) * totalArea) {
		// y-z back
		point = glm::vec4(-0.5f, c1, c2, 1);
	} else if (r < (side1 * 2 + side2 * 2 + side3) * totalArea) {
		// x-z front
		point = glm::vec4(c1, 0.5f, c2, 1);
	} else {
		// x-z back
		point = glm::vec4(c1, -0.5f, c2, 1);
	}

//	return glm::vec3(point);
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


__host__ __device__
glm::vec3 getRandomPointOnLight(Geom *g, int *lightIndices, int *lightCount, thrust::default_random_engine &rng, int& i)
{
	thrust::uniform_real_distribution<float> u01(0, *lightCount-0.001);

	int k = u01(rng);
//	if(k >= *lightCount)
//	{
//		k = *lightCount - 1;
//	}
	i = lightIndices[k];

	switch( g[i].type )
	{
		case CUBE:
			return getRandomPointOnCubeLight(g[i], rng);
		case SPHERE:
			return getRandomPointOnSphereLight(g[i], rng);
		default:
			break;
	}

	return glm::vec3(0);
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
		glm::vec3 &camPosition,
        RayState &ray,
        glm::vec3 intersect,
        glm::vec3 normal,
        Material &m,
        thrust::default_random_engine &rng,
        Geom *g,
        int geomIndex,
        int *lightIndices,
        int *lightCount)
{

	Ray &r = ray.ray;

	if(m.emittance > 0 && m.emittance < 1)
	{
		//Glowing material
		ray.rayColor *= (m.color) / m.emittance;

		//Do SSS + Fresnel split using russian roulette
		thrust::uniform_real_distribution<float> u01(0, 1);

		float split = u01(rng);

		if(split > 0.5)
		{
			//Do SSS
			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

			if(split > 0.75)
			{
				//Do Sub Surface Scattering

				//Intersect the ray with the geometry again to get a point on the geom
				Ray newR;
				newR.origin = getPointOnRay(r, m.hasTranslucence);
				newR.direction = glm::normalize(g[geomIndex].translation - newR.origin);

				glm::vec3 newIntersect, newNormal;

				if(g[geomIndex].type == SPHERE)
				{
					sphereIntersectionTest(g[geomIndex], newR, newIntersect, newNormal);
				}

				else if(g[geomIndex].type == CUBE)
				{
					boxIntersectionTest(g[geomIndex], newR, newIntersect, newNormal);
				}

				r.direction = glm::normalize(calculateRandomDirectionInHemisphere(newNormal, rng));
				r.origin = newIntersect + 0.001f * r.direction;
			}

			else
			{
				//Do diffused
				r.origin = intersect + 0.001f * r.direction;
			}
		}

		else
		{
			//Do refraction to get refracted ray dir
					glm::vec3 transmittedDir = (glm::refract(r.direction, normal, 1.0f / m.indexOfRefraction));

					float cos_t = glm::dot(transmittedDir, -normal),
							cos_i = glm::dot(-r.direction, normal);

					float r_parallel = (m.indexOfRefraction * cos_i - cos_t) / (m.indexOfRefraction * cos_i + cos_t),
						r_perpendicular = (cos_i - m.indexOfRefraction * cos_t) / (cos_i + m.indexOfRefraction * cos_t);

					if(split < 0.25f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular))
					{
						//do reflection

						ray.rayColor *= (m.color);
						r.direction = (glm::reflect(r.direction, normal));
						r.origin = intersect + 0.001f * r.direction;
					}

					else
					{
						//Do refraction
						ray.rayColor *= (m.color);
						r.direction = transmittedDir;
						r.origin = intersect + 0.001f * r.direction;

						//Intersect with the object again
						//float t;
						//bool outside;
						if(g[geomIndex].type == SPHERE)
						{
							/*t = */sphereIntersectionTest(g[geomIndex], r, intersect, normal);//, outside);
						}

						else if(g[geomIndex].type == CUBE)
						{
							/*t = */boxIntersectionTest(g[geomIndex], r, intersect, normal);//, outside);
						}

						r.direction = (glm::refract(r.direction, normal, m.indexOfRefraction));
						r.origin = intersect + 0.001f * r.direction;
					}
		}
	}

	else if(m.hasTranslucence > 0)
	{
		ray.rayColor *= (m.color) * 2.0f; // multiply by 2 as we take a 50% between
											  // diffused and SSS

		//Sub Surface Scattering
		//Do random splitting between diffused and sub surface for a better result
		thrust::uniform_real_distribution<float> u01(0, 1);

		r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

		if(u01(rng) > 0.5)
		{
			//Do Sub Surface Scattering

			//Intersect the ray with the geometry again to get a point on the geom
			Ray newR;
			newR.origin = getPointOnRay(r, m.hasTranslucence);
			newR.direction = glm::normalize(g[geomIndex].translation - newR.origin);

			glm::vec3 newIntersect, newNormal;

			if(g[geomIndex].type == SPHERE)
			{
				sphereIntersectionTest(g[geomIndex], newR, newIntersect, newNormal);
			}

			else if(g[geomIndex].type == CUBE)
			{
				boxIntersectionTest(g[geomIndex], newR, newIntersect, newNormal);
			}

			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(newNormal, rng));
			r.origin = newIntersect + 0.001f * r.direction;
		}

		else
		{
			//Do diffused
			r.origin = intersect + 0.001f * r.direction;
		}
	}

	else if(m.hasReflective == 0 && m.hasRefractive == 0)
	{
		//Diffused material
		thrust::uniform_real_distribution<float> u01(0, 1);
		if(m.specular.exponent > 0 && u01(rng) < 0.2f)
		{
			//Do specular reflection
			int i;
			glm::vec3 lightVector = glm::normalize(getRandomPointOnLight(g, lightIndices, lightCount, rng, i) - intersect);
			glm::vec3 camVector = glm::normalize(camPosition - intersect);

			float specTerm = glm::dot(normal, glm::normalize(lightVector+camVector));
			specTerm = powf(specTerm, m.specular.exponent);

			ray.rayColor *= ( m.specular.color * specTerm );

			//r.direction = glm::reflect(r.direction, normal);
			r.origin = intersect + 0.001f * r.direction;
		}

		else
		{
			//Do perfect diffused
			ray.rayColor *= (m.color);
			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
			r.origin = intersect + 0.001f * r.direction;
		}
	}

	else if (m.hasReflective > 0 && m.hasRefractive > 0)
	{
		//Do frenels reflection
		//REFERENCE : PBRT Page 435
		thrust::uniform_real_distribution<float> u01(0, 1);

		//Do refraction to get refracted ray dir
		glm::vec3 transmittedDir = (glm::refract(r.direction, normal, 1.0f / m.indexOfRefraction));

		float cos_t = glm::dot(transmittedDir, -normal),
				cos_i = glm::dot(-r.direction, normal);

		float r_parallel = (m.indexOfRefraction * cos_i - cos_t) / (m.indexOfRefraction * cos_i + cos_t),
			r_perpendicular = (cos_i - m.indexOfRefraction * cos_t) / (cos_i + m.indexOfRefraction * cos_t);

		if(u01(rng) < 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular))
		{
			//do reflection

			ray.rayColor *= (m.color);
			r.direction = (glm::reflect(r.direction, normal));
			r.origin = intersect + 0.001f * r.direction;
		}

		else
		{
			//Do refraction
			ray.rayColor *= (m.color);
			r.direction = transmittedDir;
			r.origin = intersect + 0.001f * r.direction;

			//Intersect with the object again
			//float t;
			//bool outside;
			if(g[geomIndex].type == SPHERE)
			{
				/*t = */sphereIntersectionTest(g[geomIndex], r, intersect, normal);//, outside);
			}

			else if(g[geomIndex].type == CUBE)
			{
				/*t = */boxIntersectionTest(g[geomIndex], r, intersect, normal);//, outside);
			}

			r.direction = (glm::refract(r.direction, normal, m.indexOfRefraction));
			r.origin = intersect + 0.001f * r.direction;
		}

	}

	else if(m.hasReflective > 0)
	{
		//Reflective surface
		ray.rayColor *= m.color;
		r.direction = (glm::reflect(r.direction, normal));
		r.origin = intersect + 0.001f * r.direction;
	}

	else if(m.hasRefractive > 0)
	{
		//Refractive surface
		ray.rayColor *= m.color;
		r.direction = (glm::refract(r.direction, normal, 1.0f / m.indexOfRefraction));
		r.origin = intersect + 0.001f * r.direction;

		//Intersect with the object again
//		float t;
//		bool outside;
		if(g[geomIndex].type == SPHERE)
		{
			/*t = */sphereIntersectionTest(g[geomIndex], r, intersect, normal);//, outside);
		}

		else if(g[geomIndex].type == CUBE)
		{
			/*t = */boxIntersectionTest(g[geomIndex], r, intersect, normal);//, outside);
		}

//		if (t > 0)
		{
			r.direction = (glm::refract(r.direction, normal, m.indexOfRefraction));
			r.origin = intersect + 0.001f * r.direction;
		}
	}
}
