#include "Mesh.h"

Mesh::Mesh(string filename)
{
    m_FILENAME = filename;

    buildGeomtery();
    vertexAllTriangles();
}

Mesh::~Mesh() 
{
}

void Mesh::buildGeomtery()
{
    // load the file up
    ifstream myfile(m_FILENAME);
    string line;
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            placeToken(line, &myfile);
        }
        myfile.close();
    }
    return;
}

glm::vec3 parseOneVec3(string token) {
    float x, y, z;
    sscanf_s(token.c_str(), "%f %f %f\n", &x, &y, &z);
    return glm::vec3(x, y, z);
}

void parseTwoVec3(string token, glm::vec3 *a, glm::vec3 *b)
{
    float x1, y1, z1;
    float x2, y2, z2;
    sscanf_s(token.c_str(), "%f/%f %f/%f %f/%f\n", &x1, &x2, &y1, &y2, &z1, &z2);
    *a = glm::vec3(x1, y1, z1);
    *b = glm::vec3(x2, y2, z2);
    return;
}

void Mesh::placeToken(string token, ifstream *myfile)
{
    if (token.length() == 0)
    {
        ////std::cout << "newline maybe" << std::endl;
        return;
    }

    // case of a vertex
    if (token.compare(0, 1, "v") == 0)
    {
        token.erase(0, 2); // we'll assume v x y z, so erase v and whitespace space
        vertices.push_back(parseOneVec3(token));
        return;
    }

    // case of a face index
    if (token.compare(0, 1, "f") == 0)
    {
        token.erase(0, 2); // we'll assume f x y z, so erase f and whitespace space

        // the token can be of form p1 p2 p3 or p1/t1 p2/t2 p3/t3. also, not zero-indexed
        std::size_t hasSlash = token.find_first_of('/');
        // there is no slash... just like there is no spoon oooOooooooOOOOOOoooo
        if (hasSlash == std::string::npos)
        {
            glm::vec3 face_indices = parseOneVec3(token);
            face_indices -= glm::vec3(1.0f);
            indices.push_back((int)face_indices[0]);
            indices.push_back((int)face_indices[1]);
            indices.push_back((int)face_indices[2]);
            //std::cout << "face is " << face_indices[0] << " " << face_indices[1] << " " << face_indices[2] << std::endl;

        }
        else
        {
            glm::vec3 face_indices;
            glm::vec3 tex_indices;
            parseTwoVec3(token, &face_indices, &tex_indices);

            face_indices -= glm::vec3(1.0f);
            indices.push_back((int)face_indices[0]);
            indices.push_back((int)face_indices[1]);
            indices.push_back((int)face_indices[2]);

            //std::cout << "face is " << face_indices[0] << " " << face_indices[1] << " " << face_indices[2] << " ";
            //std::cout << "tex is " << tex_indices[0] << " " << tex_indices[1] << " " << tex_indices[2] << std::endl;

            // TODO: handle the texture coordinate thing!
        }
        return;
    }
    return;
}

void Mesh::vertexAllTriangles()
{
    int numIndices = indices.size();

    // count through all the indices and push those vertices on triangleVertices;
    vector<glm::vec3> triangleVertices;
    for (int i = 0; i < numIndices; i++)
    {
        int index = indices.at(i);
        triangleVertices.push_back(vertices.at(index));
    }
    // set indices
    vector<unsigned int> triangleIndices;
    for (int i = 0; i < numIndices; i++)
    {
        triangleIndices.push_back(i);
    }

    // aaaaand switch all the buffers! CANNOT BE REVERSED
    vertices = triangleVertices;
    indices = triangleIndices;
}