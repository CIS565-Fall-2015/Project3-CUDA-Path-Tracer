#include <iostream>
#include <algorithm>
#include <list>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>


//this code only runs on cpu
AABB getAABB(const Geom & geom)
{
	AABB aabb;
	switch(geom.type)
	{
	case CUBE:
		{
			glm::vec4 tmp_arys[] = {
				glm::vec4(0.5f,0.5f,0.5f,1.0f)
				,glm::vec4(0.5f,0.5f,-0.5f,1.0f)
				,glm::vec4(0.5f,-0.5f,0.5f,1.0f)
				,glm::vec4(-0.5f,0.5f,0.5f,1.0f)
				,glm::vec4(0.5f,-0.5f,-0.5f,1.0f)
				,glm::vec4(-0.5f,0.5f,-0.5f,1.0f)
				,glm::vec4(-0.5f,-0.5f,0.5f,1.0f)
				,glm::vec4(-0.5f,-0.5f,-0.5f,1.0f)
			};
			std::vector<glm::vec4> points(&tmp_arys[0], &tmp_arys[0]+8);
			
			glm::vec4 & t = points.at(0);
			aabb.max_pos = glm::vec3(t.x/t.w,t.x/t.w,t.x/t.w);
			aabb.min_pos = glm::vec3(t.x/t.w,t.x/t.w,t.x/t.w);
			for( auto p : points)
			{
				p = geom.transform * p;
				p /= p.w;

				aabb.min_pos.x = min(aabb.min_pos.x, p.x);
				aabb.min_pos.y = min(aabb.min_pos.y, p.y);
				aabb.min_pos.z = min(aabb.min_pos.z, p.z);

				aabb.max_pos.x = max(aabb.max_pos.x, p.x);
				aabb.max_pos.y = max(aabb.max_pos.y, p.y);
				aabb.max_pos.z = max(aabb.max_pos.z, p.z);
			}
		}
		break;
	case SPHERE:
		{
			//simple square like cube
			//use max radius
			glm::vec4 tmp = geom.transform * glm::vec4(0.0f,0.0f,0.0f,1.0f);
			glm::vec3 o(tmp.x/tmp.w,tmp.y/tmp.w,tmp.z/tmp.w);
			float r = max(geom.scale.x,geom.scale.y);
			r = 0.5f * max(r,geom.scale.z);
			glm::vec3 offset(r,r,r);
			aabb.min_pos = o - offset;
			aabb.max_pos = o + offset;
		}
		break;
	case TRIANGLE:
		{
			//ugly implementation
			const glm::vec3 & a = geom.translation;
			const glm::vec3 & b = geom.rotation;
			const glm::vec3 & c = geom.scale;

			float minx = a.x;
			float miny = a.y;
			float minz = a.z;

			float maxx = a.x;
			float maxy = a.y;
			float maxz = a.z;


			minx = min(minx,b.x);
			miny = min(miny,b.y);
			minz = min(minz,b.z);
			minx = min(minx,c.x);
			miny = min(miny,c.y);
			minz = min(minz,c.z);

			maxx = max(maxx,b.x);
			maxy = max(maxy,b.y);
			maxz = max(maxz,b.z);
			maxx = max(maxx,c.x);
			maxy = max(maxy,c.y);
			maxz = max(maxz,c.z);


			aabb.min_pos = glm::vec3(minx,miny,minz);
			aabb.max_pos = glm::vec3(maxx,maxy,maxz);
		}
		break;
	default:
		std::cerr<<"GEOM TYPE ERROR\n";
		break;
	}
	return aabb;
}



std::pair<AABB,AABB> cutAABB(const AABB & parent,const AAPlane& pl)
{
	AABB l = parent;
	AABB r = parent;

	//suppose pl is always inside the parent aabb

	l.max_pos[pl.axis] = pl.pos;
	r.min_pos[pl.axis] = pl.pos;
	return std::make_pair(l,r);
}



typedef bool (*KdConstructCompareFun)(const KDNodeConstructWrapper &,const KDNodeConstructWrapper &);
bool my_kd_construct_compare_x(const KDNodeConstructWrapper & a, const KDNodeConstructWrapper & b)
{
	return a.mid.x < b.mid.x;
}
bool my_kd_construct_compare_y(const KDNodeConstructWrapper & a, const KDNodeConstructWrapper & b)
{
	return a.mid.y < b.mid.y;
}
bool my_kd_construct_compare_z(const KDNodeConstructWrapper & a, const KDNodeConstructWrapper & b)
{
	return a.mid.z < b.mid.z;
}

void KDTree::init(Scene & s)
{
	last_idx = 0;
	AABB spaceAABB;
	spaceAABB = getAABB(s.geoms[0]);

	vector<KDNodeConstructWrapper> vec_geoms(s.geoms.size());

	int i = 0;
	for(auto & g : s.geoms)
	{

		vec_geoms.at(i).aabb = getAABB(g);
		vec_geoms.at(i).geom_idx = i;
		
		AABB & aabb = vec_geoms.at(i).aabb;

		vec_geoms.at(i).mid = (aabb.max_pos + aabb.min_pos) * 0.5f;

		//update spaceAABB
		spaceAABB.min_pos.x = min(spaceAABB.min_pos.x, aabb.min_pos.x);
		spaceAABB.min_pos.y = min(spaceAABB.min_pos.y, aabb.min_pos.y);
		spaceAABB.min_pos.z = min(spaceAABB.min_pos.z, aabb.min_pos.z);

		spaceAABB.max_pos.x = max(spaceAABB.max_pos.x, aabb.max_pos.x);
		spaceAABB.max_pos.y = max(spaceAABB.max_pos.y, aabb.max_pos.y);
		spaceAABB.max_pos.z = max(spaceAABB.max_pos.z, aabb.max_pos.z);
		////////////////////


		i++;
	}

	hst_node.resize(vec_geoms.size()*2.5);

	
	root_idx = build(vec_geoms,spaceAABB,-1,0);
}

//return this node idx
int KDTree::build(vector<KDNodeConstructWrapper>& construct_objs,const AABB& box,int parent_idx,int depth)
{
	if(construct_objs.empty())
	{
		return -1;
	}


	if(last_idx >= hst_node.size())
	{
		hst_node.push_back(Node());
	}
	Node & cur = hst_node.at(last_idx);
	int cur_idx = last_idx;
	last_idx ++;


	if(construct_objs.size() <= 1)
	{
		//leaf node
		//no more split
		auto t = construct_objs.begin();
		cur.aabb = box;//t->aabb;
		cur.geom_index = t->geom_idx;
		cur.parent_idx = parent_idx;
		cur.left_idx = -1;
		cur.right_idx = -1;

		return cur_idx;
	}



	//internal node

	KdConstructCompareFun f;
	switch(depth % 3)
	{
	case 0:
		f = my_kd_construct_compare_x;
		cur.split.axis = AXIS_X;
		break;
	case 1:
		f = my_kd_construct_compare_y;
		cur.split.axis = AXIS_Y;
		break;
	case 2:
		f = my_kd_construct_compare_z;
		cur.split.axis = AXIS_Z;
		break;
	}

	
	//std::nth_element(construct_objs.begin(),construct_objs.begin()+(construct_objs.size()/2),construct_objs.end(),*f);

	sort(construct_objs.begin(),construct_objs.end(),*f);

	auto t = construct_objs.begin() + (construct_objs.size()/2);

	cur.split.pos = t->mid[cur.split.axis];
	cur.aabb = box;
	cur.geom_index = -1;
	cur.parent_idx = parent_idx;
	pair<AABB,AABB> aabb_pair = cutAABB(box,cur.split);
	
	vector<KDNodeConstructWrapper> left_objs;
	vector<KDNodeConstructWrapper> right_objs;

	left_objs.assign(construct_objs.begin(),t);
	right_objs.assign(t,construct_objs.end());

	int tmp_size = left_objs.size();

	//TODO: overlap object should be added to both branch
	for (auto o : right_objs)
	{
		if (o.aabb.min_pos[cur.split.axis] < cur.split.pos)
		{
			left_objs.push_back(o);
		}
	}

	for (int i = 0; i < tmp_size; i++)	//naive parse method....
	{
		KDNodeConstructWrapper & o = left_objs.at(i);
		if (o.aabb.max_pos[cur.split.axis] > cur.split.pos)
		{
			right_objs.push_back(o);
		}
	}




	cur.left_idx = build(left_objs,aabb_pair.first,cur_idx,depth+1);
	cur.right_idx = build(right_objs,aabb_pair.second,cur_idx,depth+1);
	//if(t == construct_objs.begin())
	//{
	//	//left_objs.assign(con
	//}
	//else
	//{
	//	
	//}

	return cur_idx;
}














Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    //if (id != geoms.size()) {
    //    cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
    //    return -1;
    //} else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

		bool isObj = false;
		string objfilename;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
			else if (strcmp(line.c_str(), "triangle") == 0)
			{
				cout << "Creating new triangle..." << endl;
				newGeom.type = TRIANGLE;
			}
			else if (strcmp(line.c_str(), "obj") == 0)
			{
				cout << "Creating from obj file ..." << endl;
				isObj = true;
			}
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
			//OBJ
			else if (strcmp(tokens[0].c_str(), "OBJFILE") == 0)
			{
				objfilename = tokens[1];
			}
			//TRIANGLE
			else if (strcmp(tokens[0].c_str(), "A") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "B") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "C") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}

            utilityCore::safeGetline(fp_in, line);
        }

		
        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

		if (!isObj)
		{
			if (newGeom.type == TRIANGLE)
			{
				glm::vec3 & a = newGeom.translation;
				glm::vec3 & b = newGeom.rotation;
				glm::vec3 & c = newGeom.scale;
				
				glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
				for (int j = 0; j < 3; j++)
				{
					newGeom.transform[0][j] = n[j];	//at a
					newGeom.transform[1][j] = n[j];	//at b
					newGeom.transform[2][j] = n[j];	//at c
				}
			}
			geoms.push_back(newGeom);
		}
		else
		{
			loadObjSimple(objfilename, newGeom.transform,newGeom.invTranspose, newGeom.materialid);
		}
        
        
		
		return 1;
    //}
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
	camera.lensRadiaus = -1.0f;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
		}
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "VIEW") == 0) {
            camera.view = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "LENS_RADIUS") == 0) {
			state.camera.lensRadiaus = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FOCAL_LENGTH") == 0) {
			state.camera.focalDistance = atof(tokens[1].c_str());
		}

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	//MY: store tan(fovy) and tan(fovx)
	camera.pixelLength = glm::vec2(2*xscaled/(float)camera.resolution.x,2*yscaled/(float)camera.resolution.y);

	//MY: calculate right vector
	camera.view = glm::normalize(camera.view);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	
	//make sure up is perpendicular to view
	camera.up = glm::cross(camera.right,camera.view);



    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}


//simple implemention
//adapted from my uc berkeley cs184 ray tracer project
void Scene::loadObjSimple(const string & objname, glm::mat4 & t, glm::mat4 & t_normal, int material_id)
{
	//fast implement
	
	//now doesn't use

	//Affine3f t_normal(t.inverse().matrix().transpose());
	//glm::mat4 t_inv = glm::inverse(t);
	//glm::mat4 t_normal = glm::inverseTranspose(t);

	vector<glm::vec3> vec_Vert;
	vector<glm::vec3> vec_Nor;


	ifstream file;
	file.open(objname);

	if (!file.is_open())
	{
		std::cout << "error: Unable to open Obj file" << std::endl;
	}
	else
	{
		string line;
		while (file.good()) {
			std::vector<std::string> split;
			std::string buf;
			std::getline(file, line);
			std::stringstream ss(line);
			while (ss >> buf) {
				split.push_back(buf);
			}
			if (split.size() == 0) {
				continue;
			}
			if (split[0][0] == '#'){
				continue;
			}
			else if (split[0] == "g")
			{
				//group...
				continue;
			}
			else if (split[0] == "s")
			{
				//ignore...
				continue;
			}
			else if (split[0] == "mtllib")
			{
				//ignore
				continue;
			}
			else if (split[0] == "usemtl")
			{
				//ignore
				continue;
			}
			else if (split[0] == "v"){
				//vertex

				float x = atof(split[1].c_str());
				float y = atof(split[2].c_str());
				float z = atof(split[3].c_str());

				vec_Vert.push_back(glm::vec3(x, y, z));

			}
			else if (split[0] == "vn"){
				//normal

				float x = atof(split[1].c_str());
				float y = atof(split[2].c_str());
				float z = atof(split[3].c_str());

				vec_Nor.push_back(glm::vec3(x, y, z));
			}
			else if (split[0] == "vt")
			{
				//texture
				//ignore now
			}
			else if (split[0] == "f")
			{
				//face

				string s;
				int split_size = split.size();
				int num_v = split_size - 1;

				int v[50];
				int vt[50];
				int vn[50];
				vn[1] = -1;

				int i;
				for (i = 1; i<split_size; i++)
				{
					std::istringstream ss(split.at(i));
					getline(ss, s, '/');
					v[i] = atoi(s.c_str());
					s = "-1";

					getline(ss, s, '/');
					vt[i] = atoi(s.c_str());	//texture
					s = "-1";

					getline(ss, s, '/');
					vn[i] = atoi(s.c_str());
					s = "-1";
				}



				for (i = 3; i <= num_v; i++)
				{
					//TODO : implement transform later

					//anti-clockwise
					glm::vec3 pa = vec_Vert.at(v[1] - 1);
					glm::vec3 pb = vec_Vert.at(v[i - 1] - 1);
					glm::vec3 pc = vec_Vert.at(v[i] - 1);

					pa = glm::vec3(t * glm::vec4(pa, 1));
					pb = glm::vec3(t * glm::vec4(pb, 1));
					pc = glm::vec3(t * glm::vec4(pc, 1));

					glm::vec3 an, bn, cn;	//normal

					if (vec_Nor.size() == 0)
					{
						an = glm::normalize (glm::cross(pb - pa, pc - pa));
						bn = an;
						cn = an;
					}
					else if (vn[1] == -1)
					{
						//normal not explicitly assigned.
						//use vertex id as normal id
						
						an = glm::vec3 (t_normal * glm::vec4(vec_Nor.at(v[1] - 1), 1));
						bn = glm::vec3 (t_normal * glm::vec4(vec_Nor.at(v[i-1] - 1), 1));
						cn = glm::vec3 (t_normal * glm::vec4(vec_Nor.at(v[i] - 1), 1));

					}
					else
					{

						an = glm::vec3(t_normal * glm::vec4(vec_Nor.at(vn[1] - 1), 1));
						bn = glm::vec3(t_normal * glm::vec4(vec_Nor.at(vn[i - 1] - 1), 1));
						cn = glm::vec3(t_normal * glm::vec4(vec_Nor.at(vn[i] - 1), 1));

					}


					//ugly implementation
					Geom triangle;
					triangle.translation = pa;
					triangle.rotation = pb;
					triangle.scale = pc;
					for (int j = 0; j < 3; j++)
					{
						triangle.transform[0][j] = an[j];
						triangle.transform[1][j] = bn[j];
						triangle.transform[2][j] = cn[j];
					}
					
					triangle.materialid = material_id;
					triangle.type = TRIANGLE;


					geoms.push_back(triangle);
				}




			}
			else
			{
				std::cout << "Warning! undefined label!\n";
				abort();
			}




		}
	}


	file.close();
}
