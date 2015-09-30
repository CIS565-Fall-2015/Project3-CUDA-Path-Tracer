#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>

#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#include "kdtree.h"

using namespace std;

//#define USE_KDTREE_FLAG


class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

	void loadObjSimple(const string & objname, glm::mat4 & t, glm::mat4 & t_normal, int material_id);
public:
    //Scene(string filename);
	void loadScene(string filename);
    //~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	//thrust::host_vector<Geom> geoms;
	//thrust::host_vector<Material> materials;
    RenderState state;


	//MY
	KDTree kdtree;

	std::vector<Geom> tmp_geoms;
};
