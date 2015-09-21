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


AABB getAABB(const Geom & geom);
std::pair<AABB,AABB> cutAABB(const AABB & parent,const AAPlane& pl);

struct Node
{

	int geom_index;	// < 0 means not leaf node
	AABB aabb;

	AAPlane split;

	//Node * child[2];
	//Node * parent;
	//int child_idx[2];
	int left_idx;
	int right_idx;
	
	
	int parent_idx;
};

class Scene;

struct KDNodeConstructWrapper
{
	AABB aabb;
	int geom_idx;

	glm::vec3 mid;
};

class KDTree
{
public:
	//attention: this will copy to gpu shared memory
	int root_idx;

	//construction runs on cpu
	void init(Scene&);
	
	vector<Node> hst_node;

	//
	int last_idx;
private:
	int build(vector<KDNodeConstructWrapper>& geoms,const AABB& box,int parent_idx,int depth);
};







class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    //~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	//thrust::host_vector<Geom> geoms;
	//thrust::host_vector<Material> materials;
    RenderState state;


	//MY
	KDTree kdtree;
};
