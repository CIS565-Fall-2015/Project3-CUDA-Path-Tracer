#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "mesh.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	int loadMesh(string objectid);
	void convertOBJMeshesToCUDAObjMeshes();
public:
    Scene(string filename);
    ~Scene();

	std::vector<ObjMesh> objmesh;
	std::vector<cuda_OBJMesh> cuda_objmeshes;
	std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
