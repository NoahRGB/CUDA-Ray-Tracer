#include "Square.h"

#include <iostream>

Square::Square() {
	vaoID = 0;
	vboID[0] = 0;
	vboID[1] = 0;
}

void Square::setSize(int width, int height) {
	this->width = width;
	this->height = height;
}

void Square::setTextureToPixels(GLubyte* pixels) {
	glBindTexture(GL_TEXTURE_2D, textureName);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
}

void Square::init(float width, float height) {
	setSize(width, height);

	if (!shader.load("Basic", "./vertShader.vert", "./fragShader.frag")) {
		std::cout << "Failed to load shader" << std::endl;
	}

	glGenTextures(1, &textureName);
	glBindTexture(GL_TEXTURE_2D, textureName);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


	// position and texture values for each vertex
	float vertices[18]; // XYZ for each vertex
	vertices[0] = -1.0; vertices[1] = 1.0; vertices[2] = 0.0;
	vertices[3] = -1.0; vertices[4] = -1.0; vertices[5] = 0.0;
	vertices[6] = 1.0; vertices[7] = -1.0; vertices[8] = 0.0;

	vertices[9] = -1.0; vertices[10] = 1.0; vertices[11] = 0.0;
	vertices[12] = 1.0; vertices[13] = 1.0; vertices[14] = 0.0;
	vertices[15] = 1.0; vertices[16] = -1.0; vertices[17] = 0.0;

	float texCoords[12];
	texCoords[0] = 0.0; texCoords[1] = 1.0;
	texCoords[2] = 0.0; texCoords[3] = 0.0;
	texCoords[4] = 1.0; texCoords[5] = 0.0;

	texCoords[6] = 0.0; texCoords[7] = 1.0;
	texCoords[8] = 1.0; texCoords[9] = 1.0;
	texCoords[10] = 1.0; texCoords[11] = 0.0;

	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);

	glGenBuffers(2, vboID);

	/////////////////////////////////////////////////////////////////////////////////
	glBindBuffer(GL_ARRAY_BUFFER, vboID[0]);
	glBufferData(GL_ARRAY_BUFFER, numVertices * 3 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
	
	GLint vertexLocation = glGetAttribLocation(shader.handle(), "in_Position");
	glEnableVertexAttribArray(vertexLocation);
	glVertexAttribPointer(vertexLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
	/////////////////////////////////////////////////////////////////////////////////
	glBindBuffer(GL_ARRAY_BUFFER, vboID[1]);
	glBufferData(GL_ARRAY_BUFFER, numVertices * 3 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);
	
	GLint textureLocation = glGetAttribLocation(shader.handle(), "in_TextureCoord");
	glEnableVertexAttribArray(textureLocation);
	glVertexAttribPointer(textureLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	/////////////////////////////////////////////////////////////////////////////////

	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

}

void Square::render() {
	glUseProgram(shader.handle());

	// todo: what is a diffuse map?
	glUniform1i(glGetUniformLocation(shader.handle(), "DiffuseMap"), 0);
	glBindTexture(GL_TEXTURE_2D, textureName);

	glBindVertexArray(vaoID);
	glDrawArrays(GL_TRIANGLES, 0, numVertices);

	glBindVertexArray(0);
	glUseProgram(0);
}