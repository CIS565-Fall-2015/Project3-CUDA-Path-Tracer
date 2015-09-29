#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"

using namespace std;

class Mesh
{
public:
    Mesh(string filename);
    ~Mesh();

    virtual void buildGeomtery();
    void placeToken(string token, ifstream *myfile);

    void vertexAllTriangles(); // change buffers so you upload every vertex for every triangle
	string m_FILENAME;
	vector<glm::vec3> vertices; // vertex buffer
	vector<unsigned int> indices; // index buffer
};