#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
	glm::vec3 color;
	int pixel_index; // the index of the pixel in the array
	bool alive; //whether or not hte thread is still running
	bool inside; // whether the ray is inside an object or not (changes ior)
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
};

//Need to have arrays for our moving geometry
struct MovingGeom {
	enum GeomType type;
	int id;
	int materialid;
	bool motionBlur; // if motion blur is enabled or not
	int frames; // the frames the geom will move over
	glm::vec3 *translations;
	glm::vec3 *rotations;
	glm::vec3 *scales;
	glm::mat4 *transforms;
	glm::mat4 *inverseTransforms;
	glm::mat4 *inverseTransposes; // Not sure how to use exactly
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec2 fov;
	bool blur;
	bool dof;
	float focalDistance;
	float apertureRadius;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};
