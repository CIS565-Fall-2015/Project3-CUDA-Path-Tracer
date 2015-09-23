
#include "Mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

Mesh::MeshEntry::~MeshEntry()
{

}

void Mesh::MeshEntry::Init(const std::vector<Vertex>& Vertices,
                           const std::vector<unsigned int>& Indices)
{
    NumIndices = static_cast<unsigned int>(Indices.size());
}


void Mesh::Clear()
{

}


void Mesh::LoadMesh(const char* Filename)
{
    // Release the previously loaded mesh (if it exists)
    Clear();

    Assimp::Importer Importer;

    //change according to requirements
    const aiScene* pScene = Importer.ReadFile(Filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);

    InitFromScene(pScene, Filename);
}

void Mesh::InitFromScene(const aiScene* pScene, const std::string& Filename)
{
    m_Entries.resize(pScene->mNumMeshes);

    // Initialize the meshes in the scene one by one
    for (unsigned int i = 0 ; i < m_Entries.size() ; i++) {
        const aiMesh* paiMesh = pScene->mMeshes[i];
        InitMesh(i, paiMesh);
    }
}

void Mesh::InitMesh(unsigned int Index, const aiMesh* paiMesh)
{
    std::vector<Vertex> Vertices;
    std::vector<unsigned int> Indices;

    const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);

    for (unsigned int i = 0 ; i < paiMesh->mNumVertices ; i++) {
        const aiVector3D* pPos = &(paiMesh->mVertices[i]);
        const aiVector3D* pNormal = paiMesh->HasNormals() ? &(paiMesh->mNormals[i]) : &Zero3D;

        Vertex v(glm::vec3(pPos->x, pPos->y, pPos->z),
                 glm::vec3(pNormal->x, pNormal->y, pNormal->z));

        Vertices.push_back(v);
    }

    for (unsigned int i = 0 ; i < paiMesh->mNumFaces ; i++) {
        const aiFace& Face = paiMesh->mFaces[i];
        assert(Face.mNumIndices == 3);
        Indices.push_back(Face.mIndices[0]);
        Indices.push_back(Face.mIndices[1]);
        Indices.push_back(Face.mIndices[2]);
    }

    m_Entries[Index].vertices = Vertices;
    m_Entries[Index].indices = Indices;

    m_Entries[Index].Init(Vertices, Indices);
}


int Mesh::getNumVertices(int index)
{
    return m_Entries[index].NumIndices;
}

std::vector<glm::vec3> Mesh::getTriangles(int index)
{
    std::vector<glm::vec3> triangleCoord;

    for(int i = 0; i<m_Entries[index].NumIndices; ++i)
    {
        triangleCoord.push_back(m_Entries[index].vertices[i].vertCoord);
    }

    return triangleCoord;
}

std::vector<glm::vec3> Mesh::getNormals(int index)
{
    std::vector<glm::vec3> normalCoord;
    glm::vec3 v1, v2, v3;

    for(int i = 0; i<m_Entries[index].NumIndices; i+=3)
    {
        v1= m_Entries[index].vertices[i].vertCoord;
        v2= m_Entries[index].vertices[i+1].vertCoord;
        v3= m_Entries[index].vertices[i+2].vertCoord;

        normalCoord.push_back(glm::normalize(glm::cross((v3-v1),(v2-v1))));
    }

    return normalCoord;
}



