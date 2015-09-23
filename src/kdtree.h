#ifndef _KDTREE_
#define _KDTREE_
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
using namespace std;

class kdtree{
public:
	kdtree *lc,*rc;
	vector<int> *verIdx,*lcVerIdx,*rcVerIdx;
	int depth;
	int mesh;//only exist when this is one triangle,store the index of indices in vertices of the triangle
	float xmax,xmin,ymax,ymin,zmax,zmin;
	float lxmax,lxmin,lymax,lymin,lzmax,lzmin;
	float rxmax,rxmin,rymax,rymin,rzmax,rzmin;
	kdtree();
	kdtree(int depth,float xmax,float xmin,float ymax,float ymin,float zmax,float zmin,vector<int> *verIdx);
	kdtree(kdtree* root);
	~kdtree();

	void createTree(vector<glm::vec3> *ver,vector<int> *idx);
	void sort(vector<glm::vec3> *ver,vector<int> *idx,int depth);
	void findBoundary(vector<glm::vec3> *ver,vector<int> *idx,int depth);
	void test(kdtree *root);
	void quickSort(int low,int high,int half,vector<glm::vec3> *ver,vector<int> *idx,int depth);
	int quickPass(int low,int high,vector<glm::vec3> *ver,vector<int> *idx,int depth);
};

#endif