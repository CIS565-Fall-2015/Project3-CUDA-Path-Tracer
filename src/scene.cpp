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
			}
			else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
				loadGeom(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
				loadCamera();
				cout << " " << endl;
			}
		}
	}
}

int Scene::loadGeom(string objectid) {
	int id = atoi(objectid.c_str());
	if (id != mgeoms.size()) {
		cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
		return -1;
	}
	else {
		cout << "Loading Geom " << id << "..." << endl;
		MovingGeom newGeom; // Switching to a MovingGeom for motion blur
		newGeom.id = id;
		string line;

		//load object type
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			if (strcmp(line.c_str(), "sphere") == 0) {
				cout << "Creating new sphere..." << endl;
				newGeom.type = SPHERE;
			}
			else if (strcmp(line.c_str(), "cube") == 0) {
				cout << "Creating new cube..." << endl;
				newGeom.type = CUBE;
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
		int numFrames = 0;
		utilityCore::safeGetline(fp_in, line);
		vector<glm::vec3> tempTranslations, tempRotations, tempScales;
		bool tempBlur;
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			if (strcmp(tokens[0].c_str(), "blur") == 0) {
				if (atoi(tokens[1].c_str()) == 1) {
					tempBlur = true;
				}
				else {
					tempBlur = false;
				}
			}
			else if (strcmp(tokens[0].c_str(), "frame") == 0) {
				numFrames++;
			}
			else if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				tempTranslations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				tempRotations.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				tempScales.push_back(glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str())));
			}
			
			utilityCore::safeGetline(fp_in, line);
		}

		numFrames++; // Create extra index for storing original info when scene is reset
		// Allocate memory for the geom arrays
		newGeom.motionBlur = tempBlur;
		newGeom.translations = (glm::vec3*)malloc(numFrames * sizeof(glm::vec3));
		newGeom.rotations = (glm::vec3*)malloc(numFrames * sizeof(glm::vec3));
		newGeom.scales = (glm::vec3*)malloc(numFrames * sizeof(glm::vec3));
		newGeom.transforms = (glm::mat4*)malloc(numFrames * sizeof(glm::mat4));
		newGeom.inverseTransforms = (glm::mat4*)malloc(numFrames * sizeof(glm::mat4));
		newGeom.inverseTransposes = (glm::mat4*)malloc(numFrames * sizeof(glm::mat4));

		// And finally you can fill them for each frame, and add it onto the list of objects
		for (int i = 0; i < numFrames - 1; i++) {
			newGeom.translations[i] = tempTranslations[i];
			newGeom.rotations[i] = tempRotations[i];
			newGeom.scales[i] = tempScales[i];
			newGeom.transforms[i] = utilityCore::buildTransformationMatrix(tempTranslations[i], tempRotations[i], tempScales[i]);
			newGeom.inverseTransforms[i] = glm::inverse(newGeom.transforms[i]);
			newGeom.inverseTransposes[i] = glm::inverseTranspose(newGeom.transforms[i]);
		}

		// Save the original twice so we have a backup for when the scene is refreshed
		newGeom.translations[2] = tempTranslations[0];
		newGeom.rotations[2] = tempRotations[0];
		newGeom.scales[2] = tempScales[0];
		newGeom.transforms[2] = utilityCore::buildTransformationMatrix(tempTranslations[0], tempRotations[0], tempScales[0]);
		newGeom.inverseTransforms[2] = glm::inverse(newGeom.transforms[0]);
		newGeom.inverseTransposes[2] = glm::inverseTranspose(newGeom.transforms[0]);

		mgeoms.push_back(newGeom);
		return 1;
	}
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
		}
		else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
			fovy = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
			state.iterations = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
			state.traceDepth = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
			state.imageName = tokens[1];
		}
	}

	string line;
	int numFrames = 0;
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "EYE") == 0) {
			camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "VIEW") == 0) {
			camera.view = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
			camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "BLUR") == 0) {
			if (atoi(tokens[1].c_str()) == 1) {
				camera.blur = true;
			}
			else {
				camera.blur = false;
			}
		}
		else if (strcmp(tokens[0].c_str(), "DOF") == 0) {
			if (atoi(tokens[1].c_str()) == 1) {
				camera.dof = true;
			}
			else {
				camera.dof = false;
			}
		}
		else if (strcmp(tokens[0].c_str(), "FD") == 0) {
			camera.focalDistance = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "AR") == 0) {
			camera.apertureRadius = atof(tokens[1].c_str());
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
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 7; i++) {
			string line;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = atof(tokens[1].c_str());
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}