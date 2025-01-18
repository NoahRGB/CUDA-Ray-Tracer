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

GLubyte* framebuffer;
CUDASphere* objects;
CUDALight* lights;
Camera cam(vec3(0.0, 0.0, 0.0), 90.0);
std::map<char, bool> keys;

double lastMouseX = 0;
double lastMouseY = 0;
bool firstMouse = true;

Window::Window(int width, int height, char* title, float fps) {
	this->width = width;
	this->height = height;
	this->fps = fps;
	this->title = title;
}

void Window::reshape(GLFWwindow* window, int width, int height) {
	//this->width = width;
	//this->height = height;

	glViewport(0.0, 0.0, width, height);
}

void Window::keyInput(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_W) keys['w'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_A) keys['a'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_S) keys['s'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_D) keys['d'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_SPACE) keys['_'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_LEFT_SHIFT) keys['|'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_I) keys['i'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_J) keys['j'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_K) keys['k'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];
	if (key == GLFW_KEY_L) keys['l'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : keys['w'];

	if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
		GLFWmonitor* monitor = glfwGetWindowMonitor(window);
		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
	}
}

void Window::mouseInput(GLFWwindow* window, double x, double y) {
	if (firstMouse) {
		lastMouseX = x;
		lastMouseY = y;
		firstMouse = false;
	}

	cam.mouseMovement(lastMouseX - x, lastMouseY - y);
	lastMouseX = x, lastMouseY = y;
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

	setup();

	glfwSetFramebufferSizeCallback(window, reshape);
	reshape(window, width, height);

	glfwSetKeyCallback(window, keyInput);
	glfwSetCursorPosCallback(window, mouseInput);
	return 0;
}

void Window::setup() {
	glClearColor(0.0, 0.0, 0.0, 0.0);

	//rt.addLight(glm::vec3(20.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.7, 0.0), 5, glm::vec3(0.0, 0.0, 0.7), 2, 10);
	//rt.addSphere(glm::vec3(0.0, 0.0, -70.0), 15.0f, { glm::vec3(1, 1, 0), 0.1, 0.9, 0.5, 200 });
	//rt.addSphere(glm::vec3(0.0, 35.0, -70.0), 15.0f, { glm::vec3(0, 1, 0), 0.1, 0.9, 0.5, 200 });
	//rt.addSphere(glm::vec3(0.0, -35.0, -70.0), 15.0f, { glm::vec3(0, 0, 1), 0.1, 0.9, 0.5, 200 });
	//rt.addSphere(glm::vec3(-300.0, 0.0, -320.0), 300.0f, { glm::vec3(1, 0, 0), 0.7, 0.5, 0.0, 200 });

	//auto time1 = std::chrono::high_resolution_clock::now();
	//rt.rayTrace();
	//auto time2 = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double, std::milli> timeTaken = time2 - time1;
	//std::cout << "Time taken: " << timeTaken.count() << std::endl

	int objectCount = 1;
	objects = new CUDASphere[objectCount];
	cudaMallocManaged((void**)&objects, objectCount * sizeof(CUDASphere));
	objects[0] = CUDASphere(vec3(0.0, 0.0, -100.0), 15.0, { vec3(1.0, 1.0, 0.0), 0.3, 0.6, 0.8, 200.0 });

	int lightCount = 1;
	lights = new CUDALight[lightCount];
	cudaMallocManaged((void**)&lights, lightCount * sizeof(CUDALight));
	//lights[0] = { vec3(20.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.7, 0.0), 5, vec3(0.0, 0.0, 0.7), 2, 10 };
	//lights[0] = { vec3(0.0, 0.0, 100.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.7, 0.0), 5, vec3(0.0, 0.0, 0.7), 2, 10 };
	lights[0] = { vec3(-20.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.7, 0.0), 5, vec3(0.0, 0.0, 0.7), 2, 10 };

	cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));

	//cam = new Camera(vec3(0.0, 0.0, 0.0), 90.0);
	cudaMallocManaged((void**)&cam, sizeof(Camera));

	cudaDeviceSynchronize();
	square.init(width, height);
	launchRayTrace();
}

void Window::launchRayTrace() {

	// 32x32 grid of 32x32 blocks
	// [32x32] x [32x32] = 1,048,576 threads
	// 1000 x 1000 pixels = 1,000,000 pixels on screen

	int N = 1000;
	dim3 dimBlock(32, 32);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

	rayTrace<<<dimGrid, dimBlock>>>(width, height, framebuffer, objects, lights, cam);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();

	square.setTextureToPixels(framebuffer);
}

void Window::run() {
	while (!glfwWindowShouldClose(window) && !glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSwapBuffers(window);
		glfwPollEvents();

		display();
	}

	delete objects;
	delete lights;
	delete framebuffer;

	glfwTerminate();
}

void Window::display() {
	if (keys['w']) {
		cam.move(cam.FORWARD);
	}
	if (keys['s']) {
		cam.move(cam.BACKWARD);
	}
	if (keys['a']) {
		cam.move(cam.LEFT);
	}
	if (keys['d']) {
		cam.move(cam.RIGHT);
	}
	if (keys['_']) {
		cam.move(cam.UP);
	}
	if (keys['|']) {
		cam.move(cam.DOWN);
	}

	//cam.pitch += mouseDeltaY * cam.getSensitivity();
	//cam.yaw += mouseDeltaX * cam.getSensitivity();
	//cam.updateDirection();

	//float targetFrameDuration = 1000 / fps;
	//auto frameStartTime = std::chrono::high_resolution_clock::now();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// ####### render #######

	launchRayTrace();
	square.render();

	// ######################

	//auto frameEndTime = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double, std::milli> frameDuration = frameEndTime - frameStartTime;
	//float sleepDuration = targetFrameDuration - (float)frameDuration.count();
	//if (sleepDuration > 0) {
	//	std::this_thread::sleep_for(std::chrono::milliseconds((int)sleepDuration));
	//}
}

