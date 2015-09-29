#include <fstream>
#include "mesh.h"
#include "utilities.h"
#include <glm/gtc/matrix_inverse.hpp>


#define MAX_FLOAT 1e6
#define MIN_FLOAT -1e6

Mesh::Mesh()
{

}

Mesh::~Mesh()
{
	
}

// for visualizing: functions
// update the normal per frame for visualization.
void Mesh::SetUniformColor(glm::vec3 color)
{
	for (unsigned int i = 0; i != m_vertices.size(); ++i)
	{
		m_colors[i] = color;
	}
}

void Mesh::ComputeNormal()
{
	// reset all the normal.
	glm::vec3 zero(0.0);
	for(std::vector<glm::vec3>::iterator n = m_normals.begin(); n != m_normals.end(); ++n)
	{
		*n = zero;
	}
	// calculate normal for each individual triangle
	unsigned int triangle_num = m_triangle_list.size() / 3;
	unsigned int id0, id1, id2;
	glm::vec3 p0, p1, p2;
	glm::vec3 normal;
	
	m_per_tri_normals.resize(triangle_num);
	
	for(unsigned int i = 0; i < triangle_num; ++i)
	{
		id0 = m_triangle_list[3 * i];
		id1 = m_triangle_list[3 * i + 1];
		id2 = m_triangle_list[3 * i + 2];

		p0 = m_vertices[id0];
		p1 = m_vertices[id1];
		p2 = m_vertices[id2];

		normal = glm::cross((p1-p0), (p2-p0));
		normal=glm::normalize(normal);

		m_per_tri_normals[i] = normal;
		

		m_normals[id0] += normal;
		m_normals[id1] += normal;
		m_normals[id2] += normal;
	}
	// re-normalize all the normals.
	for(std::vector<glm::vec3>::iterator n = m_normals.begin(); n != m_normals.end(); ++n)
	{
		if (glm::length(*n) > EPSILON) // skip if norm is a zero vector
			*n = glm::normalize(*n);
	}
}

void ObjMesh::ReadFromFile(const char* filename, float scale)
{
	m_vertices.clear();
	m_faces.clear();
	m_triangle_list.clear();
	m_colors.clear();
	m_normals.clear();
	m_textcoord.clear();
	
	m_load_success = false;
	m_load_normal=false;

	std::ifstream infile(filename);
	if(!infile.good())
	{
		printf("Error in loading file %s\n", filename);
	}
	char buffer[256];
	unsigned int ip0, ip1, ip2;
	unsigned int n0, n1, n2;
	unsigned int t0, t1, t2;
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texcoord;

	while(!infile.getline(buffer,255).eof())
	{
		buffer[255] = '\0';
		if(buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			if(sscanf_s(buffer, "v %f %f %f", &pos.x, &pos.y, &pos.z) == 3)
			{
				pos = scale * pos;
				m_vertices.push_back(pos);
			}
			else
			{
				printf("Vertex is not in desired format.\n");
				exit(0);
			}
		}
		else if (buffer[0] == 'v' && buffer[1] == 'n' && (buffer[2] == ' ' || buffer[2] == 32))
		{
			// load normals from obj file.
			if(sscanf_s(buffer, "vn %f %f %f", &normal.x, &normal.y, &normal.z) == 3)
			{
				
				m_normals.push_back(normal);
			}
			else
			{
				printf("Normal is not in desired format.\n");
				exit(0);
			}

		}
		else if (buffer[0] == 'v' && buffer[1] == 't' && (buffer[2] == ' ' || buffer[2] == 32))
		{
			// load vt from obj file
			if(sscanf_s(buffer, "vt %f %f", &texcoord.x, &texcoord.y) == 2)
			{

				m_textcoord.push_back(texcoord);
			}
			else
			{
				printf("VT is not in desired format.\n");
				exit(0);
			}

		}
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			if(sscanf_s(buffer, "f %u %u %u", &ip0, &ip1, &ip2) == 3)
			{
				m_triangle_list.push_back(--ip0);
				m_triangle_list.push_back(--ip1);
				m_triangle_list.push_back(--ip2);
				m_faces.push_back(Face(ip0, ip1, ip2));
			}
			else if(sscanf_s(buffer, "f %u//%u %u//%u %u//%u", &ip0, &n0, &ip1, &n1, &ip2, &n2) == 6)
			{
				m_triangle_list.push_back(--ip0);
				m_triangle_list.push_back(--ip1);
				m_triangle_list.push_back(--ip2);
				m_faces.push_back(Face(ip0, ip1, ip2));
			}
			else if(sscanf_s(buffer, "f %u/%u %u/%u %u/%u", &ip0, &n0, &ip1, &n1, &ip2, &n2) == 6)
			{
				m_triangle_list.push_back(--ip0);
				m_triangle_list.push_back(--ip1);
				m_triangle_list.push_back(--ip2);
				m_faces.push_back(Face(ip0, ip1, ip2));
			}
			else if(sscanf_s(buffer, "f %u//%u//%u %u//%u//%u %u//%u//%u", &ip0, &n0, &t0, &ip1, &n1, &t1, &ip2, &n2, &t2) == 9)
			{
				m_triangle_list.push_back(--ip0);
				m_triangle_list.push_back(--ip1);
				m_triangle_list.push_back(--ip2);
				m_faces.push_back(Face(ip0, ip1, ip2));
			}
			else if(sscanf_s(buffer, "f %u/%u/%u %u/%u/%u %u/%u/%u", &ip0, &n0, &t0, &ip1, &n1, &t1, &ip2, &n2, &t2) == 9)
			{
				m_triangle_list.push_back(--ip0);
				m_triangle_list.push_back(--ip1);
				m_triangle_list.push_back(--ip2);
				m_faces.push_back(Face(ip0, ip1, ip2));
			}
			else
			{
				printf("Triangle indices is not in desired format.\n");
			}
		}
	}

	unsigned int n = m_vertices.size();
	
	if (m_normals.size()!=n)
	{
		m_normals.resize(n);
	}
	
	m_colors.resize(n);


	//SetUniformColor();


	m_load_success = true;
}

void ObjMesh::WriteToFile(char* filename)
{
	// TODO: export to obj
	std::ofstream fout(filename);

	//write veritcs
	int VerticesNumber=m_vertices.size();

	for (int i=0 ;i<VerticesNumber; i++)
	{
		fout<<"v"<<" "<<m_vertices[i].x<<" "<<m_vertices[i].y<<" "<<m_vertices[i].z<<std::endl;
	}

	//write index
	int IndexNumber=m_faces.size();

	for (int i=0;i<IndexNumber;i++)
	{
		fout<<"f"<<" "<<(m_faces[i].id1+1)<<" "<<(m_faces[i].id2+1)<<" "<<(m_faces[i].id3+1)<<std::endl;
	}

	fout.close();



}

void ObjMesh::GetMeshInfo(char* info)
{
	sprintf(info, "#Verts: %d | #Tris: %d", m_vertices.size(), m_faces.size());
}

void ObjMesh::PrintMeshInfo()
{
	printf("#Verts: %d | #Tris: %d", m_vertices.size(), m_faces.size());
	cout << endl;
}

void ObjMesh::ComputeAABB()
{
	//glm::vec3 sum_of_vertices = glm::vec3(0.f);
	m_AABB.m_max_point = glm::vec3(MIN_FLOAT);
	m_AABB.m_min_point = glm::vec3(MAX_FLOAT);

	for (int i = 0; i<m_vertices.size(); i++)
	{
		//sum_of_vertices += m_vertices[i];

		//set the max point 
		m_AABB.m_max_point.x = m_vertices[i].x > m_AABB.m_max_point.x ? m_vertices[i].x : m_AABB.m_max_point.x;
		m_AABB.m_max_point.y = m_vertices[i].y > m_AABB.m_max_point.y ? m_vertices[i].y : m_AABB.m_max_point.y;
		m_AABB.m_max_point.z = m_vertices[i].z > m_AABB.m_max_point.z ? m_vertices[i].z : m_AABB.m_max_point.z;
		
		//set the min point
		m_AABB.m_min_point.x = m_vertices[i].x < m_AABB.m_min_point.x ? m_vertices[i].x : m_AABB.m_min_point.x;
		m_AABB.m_min_point.y = m_vertices[i].y < m_AABB.m_min_point.y ? m_vertices[i].y : m_AABB.m_min_point.y;
		m_AABB.m_min_point.z = m_vertices[i].z < m_AABB.m_min_point.z ? m_vertices[i].z : m_AABB.m_min_point.z;
	}
		

	m_AABB.m_center = (m_AABB.m_max_point + m_AABB.m_min_point) / 2.f;

	m_AABB.m_scale = m_AABB.m_max_point - m_AABB.m_min_point;

	//set the geom box
	m_AABB.m_box.type = CUBE;
	m_AABB.m_box.translation = m_AABB.m_center;
	m_AABB.m_box.rotation = glm::vec3(0.f);
	m_AABB.m_box.scale = m_AABB.m_scale;

	m_AABB.m_box.transform = utilityCore::buildTransformationMatrix(
		m_AABB.m_box.translation, m_AABB.m_box.rotation, m_AABB.m_box.scale);
	m_AABB.m_box.inverseTransform = glm::inverse(m_AABB.m_box.transform);
	m_AABB.m_box.invTranspose = glm::inverseTranspose(m_AABB.m_box.transform);
	m_AABB.m_box.materialid = materialid;

	
}


void ObjMesh::ApplyTransformation()
{
	for (int i = 0; i < m_vertices.size(); i++)
	{
		m_vertices[i] = glm::vec3(transform*glm::vec4(m_vertices[i], 1.f));
		m_normals[i] = glm::vec3(invTranspose*glm::vec4(m_normals[i],1.f));
	}

	for (int i = 0; i<m_per_tri_normals.size(); i++)
	{
		m_per_tri_normals[i] = glm::vec3(invTranspose*glm::vec4(m_per_tri_normals[i], 1.f));
	}
}

void ObjMesh::Init()
{
	ReadFromFile(filePath.c_str(), scale_factor);
	ComputeNormal();
	ApplyTransformation();
	ComputeAABB();
}



