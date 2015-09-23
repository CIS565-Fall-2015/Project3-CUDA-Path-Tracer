#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "kdtree.h"

enum GeomType {
    SPHERE,
    CUBE,
	MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Mesh{
	glm::vec3 *vertex,*normal;
	int *normalIdx,*textureIdx,*indices;
	int vertexNum,normalNum,indexNum;
	float radius;
	glm::vec3 center;
	kdtree *tree;
	void computeBoundingSphere(){
		float num=vertexNum,max_length=-1,length;
		glm::vec3 dis;
		center=glm::vec3(0);
		for(int i=0;i<vertexNum;i++)
			center=center+vertex[i];
		center=center/num;
		for(int i=0;i<vertexNum;i++){
			dis=vertex[i]-center;
			length=glm::length(dis);
			if(length>max_length){
				max_length=length;
			}
		}
		radius=max_length;
	}
};

struct Geom {
    enum GeomType type;
	Mesh *mesh;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
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
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};
