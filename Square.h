#pragma once

#include "Shader.h"

#include <gl/glew.h>


class Square {
private:
	unsigned int vaoID;
	unsigned int vboID[2];

	int numVertices = 6;
	
	Shader shader;

	float width;
	float height;

	GLuint textureName;


public:
	Square();
	void init(float width, float height);
	void render();

	void setSize(int width, int height);
	void setTextureToPixels(GLubyte* pixels);
};

