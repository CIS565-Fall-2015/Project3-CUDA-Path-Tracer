#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

enum GeomType {
    SPHERE,
    CUBE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct RayState
{
	Ray ray;
	int pixelIndex;
	glm::vec3 rayColor;
	bool isAlive;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    //char filename[64];
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasTranslucence;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    float focalLength;
    float aperture;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec2 fov;
    glm::vec3 H;
    glm::vec3 V;
    glm::vec3 M;
    Geom camSphere;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
    std::vector<int> lightIndices;
};

