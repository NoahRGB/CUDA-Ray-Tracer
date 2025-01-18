#pragma once

#include <string>

#include <gl/glew.h>
#include <GLFW/glfw3.h>

class Shader {
private:
	std::string name;
	GLuint vertexShader;
	GLuint fragmentShader;
	GLuint programObject;

	GLuint loadShader(const char* filename, const GLenum type);

	std::string shaderInfoLog(const GLuint shader);
	std::string programInfoLog(const GLuint program);

public:
	Shader();
	~Shader();

	GLuint handle() const { return programObject; }
	bool load(const std::string name, const char* vertexFilename, const char* fragmentFilename);
};

