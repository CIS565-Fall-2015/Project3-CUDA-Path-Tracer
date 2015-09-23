#include <iostream>
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
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
				newGeom.mesh=nullptr;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
				newGeom.mesh=nullptr;
            }
			else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
				string s;
				utilityCore::safeGetline(fp_in, s);
				cout<<"Importing "<<s<<" file"<<endl;
				newGeom.mesh=loadObj(s);
				cout<<"Importing succeed"<<endl;
				cout<<"Object with "<<newGeom.mesh->vertexNum<<" vertexes and "<<newGeom.mesh->indexNum/3<<" faces"<<endl;
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
            for (int i = 0; i < 3; i++) {
                glm::vec3 translation;
                glm::vec3 rotation;
                glm::vec3 scale;
                utilityCore::safeGetline(fp_in, line);
                tokens = utilityCore::tokenizeString(line);
                if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                    newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                    newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                    newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                }
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

Mesh *Scene::loadObj(string fileName){
	string input;
	bool usVn=false;
	Mesh *m=new Mesh();
	vector<glm::vec3> ver,nor;
	vector<int> ind,tex,norIdx;
	int index;
	ifstream inObj;
	inObj.open(fileName);
	while(inObj>>input){
		//cout<<input<<endl;
		if(input=="v"){
			glm::vec3 vec;
			inObj>>vec.x;
			inObj>>vec.y;
			inObj>>vec.z;
			ver.push_back(vec);
		}
		else if(input=="vn"){
			usVn=true;
			glm::vec3 vec;
			inObj>>vec.x;
			inObj>>vec.y;
			inObj>>vec.z;
			nor.push_back(vec);
		}
		else if(input=="vt"){}
		else if(input=="f"){
			for(int j=0;j<3;j++){
				int i=0;
				inObj>>input;
				//cout<<input<<endl;
				int count=0;
				index=0;
				while(input[count]<='9'&&input[count]>='0') count++;
				for(i=0;i<count;i++)
					index+=(input[i]-'0')*pow(10.0,count-1-i);
				ind.push_back(index-1);
				i++;

				if(i<input.size()){//texture
					int count=i;
					index=0;
					while(input[count]<='9'&&input[count]>='0') count++;
					for(;i<count;i++)
						index+=(input[i]-'0')*pow(10.0,count-1-i);
					tex.push_back(index-1);
					i++;
				}
				if(i<input.size()){//normal
					int count=i;
					index=0;
					while(input[count]<='9'&&input[count]>='0') count++;
					for(;i<count;i++)
						index+=(input[i]-'0')*pow(10.0,count-1-i);
					norIdx.push_back(index-1);
					i++;
				}
			}
		}
	}
	inObj.close();
	m->vertex=new glm::vec3[ver.size()];
	m->vertexNum=ver.size();
	for(int i=0;i<ver.size();++i) m->vertex[i]=ver[i];
	m->normal=new glm::vec3[nor.size()];
	m->normalNum=nor.size();
	for(int i=0;i<nor.size();++i) m->normal[i]=nor[i];
	m->indices=new int[ind.size()];
	m->indexNum=ind.size();
	for(int i=0;i<ind.size();++i) m->indices[i]=ind[i];
	m->computeBoundingSphere();
	initTreeStructure(m);
	return m;
}

void Scene::initTreeStructure(Mesh *m){
	float xmax,xmin,ymax,ymin,zmax,zmin;
	xmax=ymax=zmax=-1e10;xmin=ymin=zmin=1e10;
	for(int i=0;i<m->vertexNum;i++){
		if(xmax<m->vertex[i].x) xmax=m->vertex[i].x;
		if(xmin>m->vertex[i].x) xmin=m->vertex[i].x;
		if(ymax<m->vertex[i].y) ymax=m->vertex[i].y;
		if(ymin>m->vertex[i].y) ymin=m->vertex[i].y;
		if(zmax<m->vertex[i].z) zmax=m->vertex[i].z;
		if(zmin>m->vertex[i].z) zmin=m->vertex[i].z;
	}
	vector<int> *verIdx=new vector<int>;
	for(int i=0;i<m->indexNum/3;i++) verIdx->push_back(i);
	m->tree=new kdtree(0,xmax,xmin,ymax,ymin,zmax,zmin,verIdx);
	vector<glm::vec3> ver;
	vector<int> ind;
	for(int i=0;i<m->vertexNum;++i) ver.push_back(m->vertex[i]);
	for(int i=0;i<m->indexNum;++i) ind.push_back(m->indices[i]);
	m->tree->createTree(&ver,&ind);
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
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

        //load camera properties
        for (int i = 0; i < 3; i++) {
            //glm::vec3 translation; glm::vec3 rotation; glm::vec3 scale;
            utilityCore::safeGetline(fp_in, line);
            tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "EYE") == 0) {
                camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "VIEW") == 0) {
                camera.view = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
                camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

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
        for (int i = 0; i < 10; i++) {
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
