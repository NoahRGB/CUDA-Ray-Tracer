#include "Window.h"
#include "RayTracer.h"
#include "vec3.h"
#include "kernels.h"
#include "CUDASphere.h"
#include "Camera.h"

#include <gl/glew.h>
#include <GLFW/glfw3.h>

#include <chrono>
#include <thread>
#include <iostream>
#include <map>
#include <stdlib.h>


Window::Window(int width, int height, char* title, float fps) {
	this->width = width;
	this->height = height;
	this->fps = fps;
	this->title = title;
	firstMouse = true;
}

void Window::reshape(GLFWwindow* window, int width, int height) {
	Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (win) {
		win->width = width;
		win->height = height;

		win->square.setSize(width, height);
		win->rayTracer.resize(width, height);
	}

	//glViewport(0.0, 0.0, width, height);
}

void Window::keyInput(GLFWwindow* window, int key, int scancode, int action, int mods) {
	Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (win) {
		if (key == GLFW_KEY_W) win->keys['w'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['w'];
		if (key == GLFW_KEY_A) win->keys['a'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['a'];
		if (key == GLFW_KEY_S) win->keys['s'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['s'];
		if (key == GLFW_KEY_D) win->keys['d'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['d'];
		if (key == GLFW_KEY_SPACE) win->keys['_'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['_'];
		if (key == GLFW_KEY_LEFT_SHIFT) win->keys['|'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['|'];
		if (key == GLFW_KEY_I) win->keys['i'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['i'];
		if (key == GLFW_KEY_J) win->keys['j'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['j'];
		if (key == GLFW_KEY_K) win->keys['k'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['k'];
		if (key == GLFW_KEY_L) win->keys['l'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['l'];

		if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
			GLFWmonitor* monitor = glfwGetWindowMonitor(window);
			const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
		}

		if (key == GLFW_KEY_V) {
			std::cout << win->rayTracer.cam.getPosition().x() << " " << win->rayTracer.cam.getPosition().y() << " " << win->rayTracer.cam.getPosition().z() << " and " << win->rayTracer.cam.yaw << ", " << win->rayTracer.cam.pitch << std::endl;
		}
	}
}

void Window::mouseInput(GLFWwindow* window, double x, double y) {
	Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (win) {
		if (win->firstMouse) {
			win->lastMouseX = x;
			win->lastMouseY = y;
			win->firstMouse = false;
		}

		win->rayTracer.cam.mouseMovement(x - win->lastMouseX, y - win->lastMouseY);
		win->lastMouseX = x, win->lastMouseY = y;
	}
}

int Window::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, title, NULL, NULL);
	if (window == NULL) {
		std::cout << "GLFW window creation failed" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cout << " GLEW ERROR" << std::endl;
		return -1;
	}

	glfwSetWindowUserPointer(window, this);

	setup();

	glfwSetFramebufferSizeCallback(window, reshape);

	glfwSetKeyCallback(window, keyInput);
	glfwSetCursorPosCallback(window, mouseInput);
	return 0;
}

void Window::setup() {
	glClearColor(0.0, 0.0, 0.0, 0.0);

	rayTracer.init(width, height);
	square.init(width, height);

	rayTracer.launchKernel();
	square.setTextureToPixels(rayTracer.framebuffer);
}

void Window::run() {
	while (!glfwWindowShouldClose(window) && !glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSwapBuffers(window);
		glfwPollEvents();

		display();
	}
	glfwTerminate();
}

void Window::display() {
	if (keys['w']) {
		rayTracer.cam.move(rayTracer.cam.FORWARD);
	}
	if (keys['s']) {
		rayTracer.cam.move(rayTracer.cam.BACKWARD);
	}
	if (keys['a']) {
		rayTracer.cam.move(rayTracer.cam.LEFT);
	}
	if (keys['d']) {
		rayTracer.cam.move(rayTracer.cam.RIGHT);
	}
	if (keys['_']) {
		rayTracer.cam.move(rayTracer.cam.UP);
	}
	if (keys['|']) {
		rayTracer.cam.move(rayTracer.cam.DOWN);
	}
	if (keys['i']) {
		rayTracer.lights[0].position = vec3(rayTracer.lights[0].position.x(), rayTracer.lights[0].position.y(), rayTracer.lights[0].position.z() - 1);
	}
	if (keys['j']) {
		rayTracer.lights[0].position = vec3(rayTracer.lights[0].position.x() - 1, rayTracer.lights[0].position.y(), rayTracer.lights[0].position.z());
	}
	if (keys['k']) {
		rayTracer.lights[0].position = vec3(rayTracer.lights[0].position.x(), rayTracer.lights[0].position.y(), rayTracer.lights[0].position.z() + 1);
	}
	if (keys['l']) {
		rayTracer.lights[0].position = vec3(rayTracer.lights[0].position.x() + 1, rayTracer.lights[0].position.y(), rayTracer.lights[0].position.z());
	}

	// ####### render #######

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	rayTracer.launchKernel();
	square.setTextureToPixels(rayTracer.framebuffer);
	square.render();

	// ######################
}

