#pragma once

#include "sceneStructs.h"


AABB getAABB(const Geom & geom);
std::pair<AABB, AABB> cutAABB(const AABB & parent, const AAPlane& pl);

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
	int build(vector<KDNodeConstructWrapper>& geoms, const AABB& box, int parent_idx, int depth);
};
