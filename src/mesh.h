#ifndef _MESH_H_
#define _MESH_H_

#include <vector>
#include <iostream>
#include <string>
#include "glm/glm.hpp"
#include "AABB.h"


class Mesh
{
public:
	
	struct Face{
		unsigned int id1,id2,id3;
		Face() {}
		Face(int a, int b, int c) : id1(a), id2(b), id3(c){}
		void IDMinusMinus() {id1--; id2--; id3--;}
	};


	Mesh();
	virtual ~Mesh();


	// pure virtual functions
	virtual void ReadFromFile(const char* filename, float scale = 1.0f) = 0;
	virtual void WriteToFile(char* filename) = 0;
	virtual void GetMeshInfo(char* info) = 0;
	//virtual void setTexturePath(std::string filename)=0;
	
	// inline functions
	inline bool Info() {return m_load_success;}
	
	


	// for visualizing: functions
	// update the normal per frame for visualization.
	void Mesh::SetUniformColor(glm::vec3 color);
	void ComputeNormal();

	// for visualizing: data
	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec3> m_normals;
	std::vector<glm::vec3> m_colors;
	std::vector<glm::vec3> m_per_tri_normals;
	std::vector<glm::vec2> m_textcoord;
	std::vector<unsigned int> m_triangle_list;

	bool m_load_success;
	bool m_load_normal;

};

class ObjMesh :  public Mesh
{
public:
	ObjMesh() {}
	virtual ~ObjMesh() {m_faces.clear();}

	virtual void ReadFromFile(const char* filename, float scale = 1.0f);
	virtual void WriteToFile(char* filename);
	virtual void GetMeshInfo(char* info);
	
	void PrintMeshInfo();

	void ComputeAABB();
	void ApplyTransformation();
	void Init();

	std::vector<Face> m_faces;
	AABB m_AABB;
	int materialid;
	string filePath;
	float scale_factor;
	glm::vec3 m_translate;
	glm::vec3 m_rotate;
	glm::vec3 m_scale;
	glm::mat4 transform;
	glm::mat4 inverseTransform; 
	glm::mat4 invTranspose;

	
	
};


struct cuda_OBJMesh  
{
	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec3> m_normals;
	std::vector<glm::vec3> m_per_tri_normals;
	std::vector<unsigned int> m_triangle_list;
	Geom m_box;

};

struct cuda_OBJMesh_head
{
	glm::vec3* m_v;
	glm::vec3* m_n;
	unsigned int* m_tri_list;
	Geom m_box;

	int num_of_v;
	int num_of_tri;

};

#endif