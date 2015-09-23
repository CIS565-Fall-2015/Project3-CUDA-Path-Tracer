//Reference : http://ogldev.atspace.co.uk/www/tutorial22/tutorial22.html
//Library used for loading mesh into the scene

#pragma once

#ifndef __MESH__LOADER__
#define __MESH__LOADER__

#include <vector>
#include <glm/glm.hpp>
#include <assimp/scene.h>


struct Vertex
{
    glm::vec3 vertCoord;
    glm::vec3 normal;

    Vertex();
    Vertex(glm::vec3 v, glm::vec3 n)
    {
        vertCoord = v;
        normal = n;
    }
};

class Mesh
{
public:
    void LoadMesh(const char* Filename);

    int getNumVertices(int index);
    std::vector<glm::vec3> getTriangles(int index);
    std::vector<glm::vec3> getNormals(int index);

private:
    void InitFromScene(const aiScene* pScene, const std::string& Filename);
    void InitMesh(unsigned int Index, const aiMesh* paiMesh);
    void Clear();

    struct MeshEntry {
        MeshEntry() { NumIndices  = 0; }

        ~MeshEntry();

        void Init(const std::vector<Vertex>& Vertices,
                  const std::vector<unsigned int>& Indices);


        unsigned int NumIndices;

        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
    };

    std::vector<MeshEntry> m_Entries;
};

#endif
