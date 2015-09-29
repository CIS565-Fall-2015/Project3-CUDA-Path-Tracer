#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {//generate a random number
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
__host__ __device__ float R_coeff(float theta){
	//air :n1=1; ice:1.3
	float n1, n2;
	n1 = 1.0; n2 = 1.3;
	float R0 = pow((n1 - n2) / (n1 + n2), 2);
	float Rtheta = R0 + (1 - R0)*pow((1 - theta), 5);
	return Rtheta;
}
/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * *   
 * The visual effect you want is to straight-up add the diffuse and specular  
 * components. You can do this in a few ways:  
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
__host__ __device__ void reflectRay(Ray &r,glm::vec3 normal,glm::vec3 intersection,Material m){
	r.direction = glm::reflect(r.direction, normal);
	r.origin = intersection;
	r.hitcolor *= m.specular.color;
}
__host__ __device__ void refractRay(Ray &r, glm::vec3 normal, glm::vec3 intersection, Material m){
	float n = 1.0 / 1.3;//n1/n2(air to ice)
	//r.direction = glm::refract(r.direction, normal);
	r.origin = intersection;
	r.hitcolor *= m.specular.color;
}
__host__ __device__
void scatterRay(Ray &ray,glm::vec3 &color,glm::vec3 intersect,glm::vec3 normal,glm::vec3 emittedColor,const Material &m,
                thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	
	if (m.hasRefractive&&m.hasReflective){

		float theta = glm::dot(ray.direction, normal);
		float Rtheta = R_coeff(theta);
		thrust::uniform_real_distribution <float> u01(0, 1);
		glm::vec3 refract_ray,reflect_ray;
		if (u01(rng)< Rtheta / (1 - Rtheta))//refract
		{
			//ray.direction = glm::refract(ray.direction, normal, 1.0);
		}
		else{
			ray.direction = glm::reflect(-ray.direction, normal);
		    ray.origin = intersect;
		//	float Specular = glm::pow(Base, Material.exponent()); 
		//	color = m.specular.color * Specular
		
		
	}
	
	//diffuse
	if ((!m.hasReflective) && (!m.hasRefractive)){
		glm::vec3 diffuse_ray = calculateRandomDirectionInHemisphere(normal, rng);
		ray.direction = glm::normalize(diffuse_ray);
		ray.origin = intersect;
		color = m.color;
	}
	//else color 
}
