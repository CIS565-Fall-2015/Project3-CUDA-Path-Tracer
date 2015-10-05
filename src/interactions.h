#pragma once

#include "sceneStructs.h"
#include "intersections.h"

__device__ glm::vec3 localToWorldTransform(glm::vec3 local,
        float up, float over, float around) {

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    glm::vec3 otherDirection;
    if (abs(local.x) < SQRT_OF_ONE_THIRD) {
        otherDirection = glm::vec3(1, 0, 0);
    } else if (abs(local.y) < SQRT_OF_ONE_THIRD) {
        otherDirection = glm::vec3(0, 1, 0);
    } else {
        otherDirection = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(local, otherDirection));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(local, perpendicularDirection1));

    return up * local
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__ glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
    return localToWorldTransform(normal, up, over, around);
}

__device__ glm::vec3 sampleSpecular(Ray &ray, glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u(-1, 1);
    glm::vec3 specular = glm::reflect(ray.direction, normal);

    float x1 = u(rng);
    float theta = glm::acos(glm::pow(x1, 1/(m.specular.exponent+1)));
    float x2 = u(rng);
    float phi = x2 * TWO_PI;

    float over = glm::cos(phi) * glm::sin(theta);
    float around = glm::sin(phi) * glm::sin(theta);
    float up = glm::cos(theta);

    return localToWorldTransform(specular, up, over, around);
}

__device__ glm::vec3 sampleTexture(Texture t, glm::vec2 uv) {
    int x = glm::clamp(uv.x, 0.f, 1.f) * (t.width - 1);
    int y = glm::clamp(uv.y, 0.f, 1.f) * (t.height - 1);
    x %= t.width;
    y %= t.height;
    int i = (x + (y * t.width))*t.channels;
    unsigned char* data = t.data;
    glm::vec3 sample = glm::vec3(data[i], data[i+1], data[i+2]) / 255.f;
    return sample;
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
__device__ void scatterRay(
        Ray &ray, glm::vec3 &color,
        glm::vec3 intersect, glm::vec3 normal, glm::vec2 uv, bool outside,
        const Material &m,
        thrust::default_random_engine &rng) {

    if (m.normalid >= 0) {
        normal = sampleTexture(m.normalMap, uv);
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    if (m.indexOfRefraction != -1) {
        float n1 = outside  ? 1.0f : m.indexOfRefraction;
        float n2 = !outside ? 1.0f : m.indexOfRefraction;
        float r0 = (n1 - n2) / (n1 + n2);
        r0 = r0 * r0;

        float cos_theta = glm::dot(normal, -1.f * ray.direction);
        float schlick = r0 + (1.f - r0) * glm::pow(1.f - cos_theta, 5);

        if (rand < schlick) {
            // Specular
            glm::vec3 specular = sampleSpecular(ray, normal, m, rng);
            color *= m.specular.color;
            ray.origin = intersect;
            ray.direction = specular;
        } else {
            // Refract
            float ior = n1 / n2;
            color *= m.color / (1-schlick);
            ray.origin = intersect + (ray.direction * 0.001f);
            ray.direction = glm::refract(ray.direction, normal, ior);
        }
    } else {
        glm::vec3 spec = m.specular.color;
        glm::vec3 diff = m.textureid >= 0 ? sampleTexture(m.texture, uv) : m.color;
        float specularIntensity = spec.x + spec.y + spec.z;
        float diffIntensity     = diff.x + diff.y + diff.z;
        float totalIntensity    = specularIntensity + diffIntensity;
        float specularProb = specularIntensity / totalIntensity;
        float diffProb = diffIntensity / totalIntensity;

        if (rand < specularProb) {
            // Specular
            glm::vec3 specular = sampleSpecular(ray, normal, m, rng);
            color *= m.specular.color / specularProb;
            ray.origin = intersect;
            ray.direction = specular;
        } else {
            // Diffuse
            color *= diff / diffProb;
            ray.origin = intersect;
            ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        }
    }

    if (m.emittance > 0) {
        color *= m.emittance;
    }
}
