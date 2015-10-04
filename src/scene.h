#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadGeom(string objectid);
    int loadMaterial(string materialid);
    int loadTexture(const char *filename);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    RenderState state;
};
