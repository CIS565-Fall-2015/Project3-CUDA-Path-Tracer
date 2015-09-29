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
//http://www.igorsklyar.com/system/documents/papers/4/fiscourse.comp.pdf
__host__ __device__
glm::vec3 calculateRandomSpecularDirection(
glm::vec3 specDir,float specExp, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float X1 = u01(rng);
	float X2 = u01(rng);
	float n = specExp;

	float thi = acos(pow(X1, 1 / (n+1)));
	float phi = TWO_PI*X2;

	float z = cos(thi);
	float x = cos(phi)*sin(thi);
	float y = sin(phi)*sin(thi);

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(specDir.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(specDir.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(specDir, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(specDir, perpendicularDirection1));

	return z * specDir
		+ x * perpendicularDirection1
		+ y * perpendicularDirection2;
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
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__host__ __device__ glm::vec3 ColorInTex(int texId,glm::vec3**texs,glm::vec2*info,int x,int y)
{
	int xSize = info[texId].x;
	int ySize = info[texId].y;
	if (x < 0 || y < 0 || x >= xSize || y >= ySize) return glm::vec3(0, 0, 0);
	return texs[texId][(y * xSize) + x];
}

__host__ __device__
glm::vec3 scatterRay(
Ray &ray,
bool outside,
float intrT,
glm::vec3 intersect,
glm::vec3 normal,
const Material &m,
glm::vec3**t,
glm::vec2*info,
thrust::default_random_engine &rrr) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.
	enum RayType
	{
		DiffRay, // Diffuse ray : calculateRandomDirectionInHemisphere
		ReflRay, // (Mirror) Reflected ray: glm::reflect(ray.direction, normal);
		RefrRay, // (Transparent) refracted ray : n1/n2
		SpecRay, // (Non-perfect Mirror) calculateRandomSpecularDirection
		SSSRay_o,   //
		SSSRay_i
	};
	int ray_type;
	float specProb = 0;
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (ray.terminated)
		return ray.carry;

	glm::vec3 matColor = m.color;
	if (m.TexIdx!=-1)
	{
		matColor = ColorInTex(m.TexIdx,t,info, 700,100);
		//matColor = glm::vec3(0, 1, 0);
	}

	if (m.emittance > 0)
	{
		ray.carry *= m.emittance*matColor;// m.color;
		ray.terminated = true;
		return ray.carry;
	}
	// Shading 
	else if (m.hasRefractive)
	{//later
		float cos_thi = glm::dot(glm::normalize(-ray.direction), glm::normalize(normal));
		float R0 = (m.indexOfRefraction-1) / (1+m.indexOfRefraction);
		R0 *= R0;
		float R_thi = R0 + (1 - R0)*pow(1 - cos_thi, 5);
		if (outside)
		{
			if (u01(rrr)<R_thi) ray_type = ReflRay; //fresnel
			else ray_type = RefrRay;
		}
		else
			ray_type = RefrRay;

	}
	else if (m.hasReflective)
	{
		if (m.bssrdf>0)
		{

			if (outside) //from outside
			{
				float cos_thi = glm::dot(glm::normalize(-ray.direction), glm::normalize(normal));
				float R_thi = pow(1 - cos_thi, 5);
				if (u01(rrr) < R_thi)//later: what value to choose?
					ray_type = ReflRay;
				else
					ray_type = SSSRay_o;
			}
			else//inside
				ray_type = SSSRay_i;
		}
		else
			ray_type = ReflRay;
	}
	else if (m.bssrdf > 0) //subsurface scattering : try brute-force bssrdf
	{
		//http://noobody.org/bachelor-thesis.pdf
		//(1) incident or exitant or currently inside obj ?
		//	if incident:
		if (outside) //from outside
		{
			if (u01(rrr) > -0.1)//later: what value to choose?
				ray_type = SSSRay_o;
			else
				ray_type = DiffRay;
		}
		else//inside
			ray_type = SSSRay_i;

	}
	else // diffuse / specular
	{
		//!!! later : specular
		//http://www.tomdalling.com/blog/modern-opengl/07-more-lighting-ambient-specular-attenuation-gamma/

		if (m.specular.exponent > 0)
		{	
			specProb = glm::length(m.specular.color);
			//specProb = specProb / (specProb + glm::length(m.color));
			specProb = specProb / (specProb + glm::length(matColor));
			if (u01(rrr) < specProb) //spec ray
				ray_type = SpecRay;
			else//diffuse ray
				ray_type = DiffRay;			
		}
		else 
			ray_type = DiffRay;
	}


	switch (ray_type)
	{
	case DiffRay:
	{
		ray.origin = getPointOnRay(ray, intrT);
		ray.carry *= matColor;//m.color;// *(1.f / (1 - specProb));
		ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rrr));
	}
		break;
	case ReflRay:
	{
		ray.origin = getPointOnRay(ray, intrT);
		ray.direction = glm::normalize(glm::reflect(ray.direction, normal));
		ray.carry *= m.specular.color;
	}
		break;
	case RefrRay:
	{
		
		float n = m.indexOfRefraction;
		if (outside)
			n = 1 / n;
		float angle = 1.0f - glm::pow(n, 2) * (1.0f - glm::pow(glm::dot(normal, ray.direction), 2));
		if (angle < 0)
		{
			ray.origin = getPointOnRay(ray, intrT);
			ray.direction = glm::normalize(glm::reflect(ray.direction, normal));
			ray.carry *= m.specular.color;
		} 
		else
		{
			ray.origin = getPointOnRay(ray, intrT + 0.001f);
			ray.direction = glm::normalize(glm::refract(ray.direction, normal, n));
			ray.carry *= matColor;// m.color;
		}
	}
		break;
	case SpecRay:
	{
		ray.origin = getPointOnRay(ray, intrT);
		glm::vec3 specDir = glm::reflect(ray.direction, normal);
		ray.direction = glm::normalize(calculateRandomSpecularDirection(specDir, m.specular.exponent, rrr));
		ray.carry *= m.specular.color;// *(1.f / specProb);
	}
		break;
	case SSSRay_o:
	{
		ray.origin = getPointOnRay(ray, intrT + .0002f);
		glm::vec3 refraDir = glm::normalize(calculateRandomDirectionInHemisphere(-normal, rrr));
		ray.carry *= matColor;// m.color;
		ray.direction = refraDir;
	}
		break;
	case SSSRay_i:
	{
		//Sigma_a: Absorption coefficient
		//Sigma_s: Scattering coefficient
		// Extinction coefficient Sigma_t = Sigma_s+Sigma_a
		float Sigma_t = m.bssrdf;
		float so = -log(u01(rrr)) / Sigma_t;
		float si = glm::length(getPointOnRay(ray, intrT) - ray.origin);
		if (si <= so) //turns into exitant, go out of the objects
		//if (true)
		{
			//ray.carry *= m.color;
			ray.origin = getPointOnRay(ray, intrT + .0002f);
			ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(-normal, rrr));
		}
		else //stays in the obj, pick new direction and scatter distance
		{
			//ray.carry *= m.color;
			ray.origin = getPointOnRay(ray, so);
			ray.direction = -glm::normalize(calculateRandomDirectionInHemisphere(ray.direction, rrr));
		}
	}
		break;
	default:
		break;
	}
	return ray.carry;

}
