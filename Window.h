#pragma once

#include <gl/glew.h>
#include <GLFW/glfw3.h>

#include "Square.h"
#include "RayTracer.h"

class Window {
private:
	int width, height;
	float fps;
	char* title;
	GLFWwindow* window;

	Square square;

	void display();
	void setup();
	static void reshape(GLFWwindow* window, int width, int height);
	static void keyInput(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouseInput(GLFWwindow* window, double x, double y);
	static void mouseButtonInput(GLFWwindow* window, int button, int action, int mods);


public:
	Window(int width, int height, char* title, float fps=60.0);

	int init();
	void run();
	void launchRayTrace();
};