#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "Mesh.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	int loadObj(string filename, Geom &geom);

public:
    Scene(string filename);
    ~Scene();
	int loadAllObjs();

	std::vector<string> filenames;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
