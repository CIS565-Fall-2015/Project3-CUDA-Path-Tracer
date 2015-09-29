#pragma once

#include "sceneStructs.h"


AABB getAABB(const Geom & geom);
std::pair<AABB, AABB> cutAABB(const AABB & parent, const AAPlane& pl);

struct Node
{
	//first geom index
	int geom_index;	// < 0 means not leaf node
	
	AABB aabb;

	AAPlane split;

	
	//left_idx = cur_idx + 1;

	int num_geoms;


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
	const int MAX_LEAF_GEOM_NUM = 5;
	const float MAX_OVERLAP_RATIO = 0.5f;


	//attention: this will copy to gpu shared memory
	int root_idx;

	//construction runs on cpu
	void init(Scene&);

	vector<Node> hst_node;

	vector<int> hst_geom_idx;
	//
	int last_idx;
private:
	int build(vector<KDNodeConstructWrapper> & geoms
		,vector<int> & sequence, const AABB& box, int parent_idx, int depth);

	void buildLeaf(int cur_idx,const vector<KDNodeConstructWrapper>& geoms
		, vector<int> & sequence, const AABB& box, int parent_idx, int depth);
};
