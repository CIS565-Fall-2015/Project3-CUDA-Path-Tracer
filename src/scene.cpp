#include <iostream>
#include <algorithm>
#include <list>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>




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
