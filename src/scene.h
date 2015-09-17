#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	//thrust::host_vector<Geom> geoms;
	//thrust::host_vector<Material> materials;
    RenderState state;
};
