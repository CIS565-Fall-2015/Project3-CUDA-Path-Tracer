#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>

#include "image.h"

image::image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new glm::vec3[x * y]),
		isTex(false){
}

image::~image() {
	if (!isTex)
	{
		delete pixels;
	}
    
}

void image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

glm::vec3 image::getPixel(int x, int y) {
	assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
	return pixels[(y * xSize) + x]; 
}

void image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

void image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) { 
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}
image::image(const std::string &baseFilename)
{
	//unsigned char *bytes = new unsigned char[3 * xSize * ySize];
	int n = 3,x,y;
	unsigned char *bytes = stbi_load(baseFilename.c_str(), &x, &y, &n, 0);
	xSize = x;
	ySize = y;
	pixels = new glm::vec3[xSize * ySize];

	for (int y = 0; y < ySize; y++) {
		for (int x = 0; x < xSize; x++) {
			int i = y * xSize + x;
			pixels[i].x = (float)bytes[3 * i + 0] / 255.f;
			pixels[i].y = (float)bytes[3 * i + 1] / 255.f;
			pixels[i].z = (float)bytes[3 * i + 2] / 255.f;
		}
	}
	isTex = true;
	//setPixel(,);
	std::cout << "loaded " << baseFilename << "." << std::endl;
	delete[] bytes;
}
void image::loadPNG(const std::string &baseFilename) {
	//unsigned char *bytes = new unsigned char[3 * xSize * ySize];
	int n = 3;
	unsigned char *bytes = stbi_load(baseFilename.c_str(), &xSize, &ySize, &n, 0);
	pixels = new glm::vec3[xSize * ySize];

	for (int y = 0; y < ySize; y++) {
		for (int x = 0; x < xSize; x++) {
			int i = y * xSize + x;
			//glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
			//bytes[3 * i + 0] = (unsigned char)pix.x;
			//bytes[3 * i + 1] = (unsigned char)pix.y;
			//bytes[3 * i + 2] = (unsigned char)pix.z;
			pixels[i].x = (float)bytes[3 * i + 0]/255.f;
			pixels[i].y = (float)bytes[3 * i + 1] / 255.f;
			pixels[i].z = (float)bytes[3 * i + 2] / 255.f;
		}
	}

	//std::string filename = baseFilename + ".png";
	//stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
	std::cout << "loaded " << baseFilename << "." << std::endl;

	delete[] bytes;
}
int image::getSize()
{
	return xSize*ySize;
}
