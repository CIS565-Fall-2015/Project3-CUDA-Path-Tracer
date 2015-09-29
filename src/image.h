#pragma once

#include <glm/glm.hpp>

using namespace std;

class image {
private:

   

public:
	int xSize;
    int ySize;
	glm::vec3 *pixels;
    image(int x, int y);
	image(const std::string &baseFilename);
    ~image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
	glm::vec3 getPixel(int x, int y);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
	void loadPNG(const std::string &baseFilename);
	int getSize();
	bool isTex;
};

