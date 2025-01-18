#include "Shader.h"

#include <iostream>
#include <fstream>

Shader::Shader() {
	name = "";
	vertexShader = 0;
	fragmentShader = 0;
	programObject = 0;
}

Shader::~Shader() {
	// cleanup
	glUseProgram(0);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteProgram(programObject);
}

GLuint Shader::loadShader(const char* filename, const GLenum type) {
	std::string source; // holds shader text
	std::ifstream file(filename, std::ios::binary);

	if (file.is_open()) {
		file.seekg(0, std::ios::end);
		
		unsigned int fileSize = static_cast<unsigned int>(file.tellg());
		source.resize(fileSize);
		file.seekg(0, std::ios::beg);
		file.read(&source[0], fileSize);
	}

	GLuint shader = glCreateShader(type);

	// todo: "reinterpret cast used to convert between pointer types"
	const GLchar* glSource = reinterpret_cast<const GLchar*>(source.c_str());
	glShaderSource(shader, 1, &glSource, 0);

	return shader;
}

bool Shader::load(const std::string name, const char* vertexFilename, const char* fragmentFilename) {
	this->name = name;
	GLint successCheck = 0; // store status at various points

	// load vertex shader
	vertexShader = loadShader(vertexFilename, GL_VERTEX_SHADER);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &successCheck);
	if (!successCheck) {
		std::cout << "Error compiling GLSL vertex shader: '" << vertexFilename << "'" << std::endl;
		return false;
	}

	// load fragment shader
	fragmentShader = loadShader(fragmentFilename, GL_FRAGMENT_SHADER);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &successCheck);
	if (!successCheck) {
		std::cout << "Error compiling GLSL fragment shader: '" << fragmentFilename << "'" << std::endl;
		return false;
	}

	// create program and link shaders
	programObject = glCreateProgram();
	glAttachShader(programObject, vertexShader);
	glAttachShader(programObject, fragmentShader);
	glLinkProgram(programObject);
	glGetProgramiv(programObject, GL_LINK_STATUS, &successCheck);
	if (!successCheck) {
		std::cout << "Error linking GLSL shaders into a shader program." << std::endl;
		std::cout << "GLSL vertex shader: '" << vertexFilename << "'" << std::endl;
		std::cout << "GLSL fragment shader: '" << fragmentFilename << "'" << std::endl;
		std::cout << "Program info log:" << std::endl << programInfoLog(programObject) << std::endl;
		return false;
	}

	std::cout << "Loaded GLSL program: " << name << std::endl;
	return true;
}

std::string Shader::shaderInfoLog(const GLuint shader) {
	GLsizei infoLogSize = 0;
	std::string infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogSize);
	infoLog.resize(infoLogSize);
	glGetShaderInfoLog(shader, infoLogSize, &infoLogSize, &infoLog[0]);

	return infoLog;
}

std::string Shader::programInfoLog(const GLuint program) {
	GLsizei infoLogSize = 0;
	std::string infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogSize);
	infoLog.resize(infoLogSize);
	glGetProgramInfoLog(program, infoLogSize, &infoLogSize, &infoLog[0]);

	return infoLog;
}