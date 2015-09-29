#pragma once

#include <glm/glm.hpp>
#include <vector>

using namespace std;

class image {
public:
    int xSize;
    int ySize;
    std::vector<glm::vec3> pixels;

    image(int x, int y);
	~image();
	void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};
