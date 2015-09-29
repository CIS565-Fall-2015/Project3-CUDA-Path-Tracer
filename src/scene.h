#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "device_launch_parameters.h"
#include "image.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
	int loadGeom(string objectid);
	int loadTexture(string textureid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
	std::vector<Material> materials;
	std::vector<image> textures;
    RenderState state;
};
