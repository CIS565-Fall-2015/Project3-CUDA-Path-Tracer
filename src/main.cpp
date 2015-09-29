#include "main.h"
#include "preview.h"
#include <cstring>
#include "glm/gtx/intersect.hpp"

static std::string startTimeString;
static bool camchanged = false;
static float theta = 0, phi = 0;
static glm::vec3 cammove;

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------


float main_triangle_intersection(glm::vec3 ori,glm::vec3 dir , glm::vec3 &v1, glm::vec3 &v2, glm::vec3 &v3)
{

	



	glm::vec3 e1, e2;
	e1 = v2 - v1;
	e2 = v3 - v1;

	glm::vec3 P = glm::cross(dir, e2);
	float det = glm::dot(e1, P);

	float m_EPSILON = 1.e-6;
	if (det > -m_EPSILON && det < m_EPSILON)
	{
		return -1;
	}

	float inv_det = 1.f / det;

	glm::vec3 T = ori - v1;

	float u = glm::dot(T, P)*inv_det;

	if (u<0.f || u>1.f)
	{
		return -1;
	}

	glm::vec3 Q = glm::cross(T, e1);

	float v = glm::dot(dir, Q)*inv_det;

	if (v<0.f || u + v>1.f)
	{
		return -1;
	}

	float t = glm::dot(e2, Q)*inv_det;

	//return t;
	if (t > m_EPSILON)
	{
		return t;
	}


	return -1;




}


int main(int argc, char** argv) {
    
	////test 
	//glm::vec3 a(3.f, 0.f, 1.f);
	//glm::vec3 c(3.f, 0.f, -1.f);
	//glm::vec3 b(3.f, 1.f, 0.f);

	//glm::vec3 ori(0.f,0.f,0.f);
	//glm::vec3 dir(1.f,0.f,0.f);
	//


	//float tmp_t = main_triangle_intersection(ori,dir, a, b, c);

	//cout << tmp_t << endl;

	//glm::vec3 bary_coord;
	//bool is_intersect = glm::intersectRayTriangle(ori, dir, a, b, c, bary_coord);
	//
	//cout << is_intersect << bary_coord.x << bary_coord.y << bary_coord.z << endl;
	//
	
	startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

	//std::cout << scene->state.camera.position.x << " " << scene->state.camera.position.y << " " << scene->state.camera.position.z << " " << std::endl;
    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    width = renderState->camera.resolution.x;
    height = renderState->camera.resolution.y;

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
        glm::vec3 v = cam.view;
        glm::vec3 u = cam.up;
        glm::vec3 r = glm::cross(v, u);
        glm::mat4 rotmat = glm::rotate(theta, r) * glm::rotate(phi, u);
        cam.view = glm::vec3(rotmat * glm::vec4(v, 0.f));
        cam.up = glm::vec3(rotmat * glm::vec4(u, 0.f));
        cam.position += cammove.x * r + cammove.y * u + cammove.z * v;
        theta = phi = 0;
        cammove = glm::vec3();
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations) {
        uchar4 *pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    } else {
        saveImage();
        pathtraceFree();
		cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            saveImage();
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_SPACE:
            saveImage();
            break;
        case GLFW_KEY_DOWN:  camchanged = true; theta = -0.1f; break;
        case GLFW_KEY_UP:    camchanged = true; theta = +0.1f; break;
        case GLFW_KEY_RIGHT: camchanged = true; phi = -0.1f; break;
        case GLFW_KEY_LEFT:  camchanged = true; phi = +0.1f; break;
        case GLFW_KEY_A:     camchanged = true; cammove -= glm::vec3(.1f, 0, 0); break;
        case GLFW_KEY_D:     camchanged = true; cammove += glm::vec3(.1f, 0, 0); break;
        case GLFW_KEY_W:     camchanged = true; cammove += glm::vec3(0, 0, .1f); break;
        case GLFW_KEY_S:     camchanged = true; cammove -= glm::vec3(0, 0, .1f); break;
        case GLFW_KEY_R:     camchanged = true; cammove += glm::vec3(0, .1f, 0); break;
        case GLFW_KEY_F:     camchanged = true; cammove -= glm::vec3(0, .1f, 0); break;
        }
    }
}
