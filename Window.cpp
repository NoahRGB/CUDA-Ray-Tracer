#include "Window.h"
#include "RayTracer.h"
#include "vec3.h"
#include "kernels.h"
#include "CUDASphere.h"
#include "Camera.h"
#include "utils.h"

#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <chrono>
#include <thread>
#include <iostream>
#include <map>
#include <stdlib.h>


Window::Window(int width, int height, char* title) {
	this->width = width;
	this->height = height;
	this->title = title;
	rayTracer.config.fps = 0;

	mouseEnabled = true;
	firstMouse = true;
}

void Window::reshape(GLFWwindow* window, int width, int height) {
	Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (win) {

		// update width & height for window, square and raytracer
		win->width = width;
		win->height = height;

		win->square.setSize(width, height);
		win->rayTracer.resize(width, height);

		glViewport(0.0, 0.0, width, height);
	}
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
			std::cout << win->rayTracer.scene.cam.getPosition().x() << " " << win->rayTracer.scene.cam.getPosition().y() << " " << win->rayTracer.scene.cam.getPosition().z() << " and " << win->rayTracer.scene.cam.yaw << ", " << win->rayTracer.scene.cam.pitch << std::endl;
		}

		if (key == GLFW_KEY_P && action == GLFW_PRESS) {
			if (!win->mouseEnabled) {
				win->mouseEnabled = true;
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else {
				win->mouseEnabled = false;
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}
}

void Window::mouseInput(GLFWwindow* window, double x, double y) {
	Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (win) {
		if (win->mouseEnabled) {
			if (win->firstMouse) {
				win->lastMouseX = x;
				win->lastMouseY = y;
				win->firstMouse = false;
			}
			win->rayTracer.scene.cam.mouseMovement(x - win->lastMouseX, y - win->lastMouseY);
			win->lastMouseX = x, win->lastMouseY = y;
		}
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

	rayTracer.scene.cam.updateDirection();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();

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
	savedTime = glfwGetTime();
	frameCount = 0;
	while (!glfwWindowShouldClose(window) && !glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSwapBuffers(window);
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Ray Tracing");
		ImGui::Text("FPS: %d", rayTracer.config.fps);

		ImGui::SeparatorText("Lighting");
		ImGui::Checkbox("Hard Shadows", &rayTracer.config.renderHardShadows);
		ImGui::Checkbox("Ambient lighting", &rayTracer.config.ambientLighting);
		ImGui::Checkbox("Diffuse lighting", &rayTracer.config.diffuseLighting);
		ImGui::Checkbox("Specular lighting", &rayTracer.config.specularLighting);
		ImGui::Checkbox("Reflections", &rayTracer.config.reflections);
		ImGui::SliderFloat("Shadow bias", &rayTracer.config.shadowBias, 0.0, 15.0);

		ImGui::SeparatorText("Create sphere");
		static float posX = 0.0; static float posY = 0.0; static float posZ = 0.0; static float radius = 20.0;
		static float r = 1.0; static float g = 0.0; static float b = 0.0;
		static float ambient = 0.1; static float diffuse = 0.9; static float specular = 0.5;
		static bool reflective = false;
		ImGui::InputFloat("PosX", &posX);
		ImGui::InputFloat("PosY", &posY);
		ImGui::InputFloat("PosZ", &posZ);
		ImGui::InputFloat("Radius", &radius);
		ImGui::InputFloat("Red", &r);
		ImGui::InputFloat("Green", &g);
		ImGui::InputFloat("Blue", &b);
		ImGui::InputFloat("Ambient", &ambient);
		ImGui::InputFloat("Diffuse", &diffuse);
		ImGui::InputFloat("Specular", &specular);
		ImGui::Checkbox("Reflective?", &reflective);
		if (ImGui::Button("Create Sphere")) {
			rayTracer.addSphere(vec3(posX, posY, posZ), radius, { vec3(r, g, b), ambient, diffuse, specular, 200.0 }, reflective ? Reflect : Diffuse);
		}

		ImGui::End();

		display();
	}
	glfwTerminate();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Window::display() {
	if (keys['w']) {
		rayTracer.scene.cam.move(rayTracer.scene.cam.FORWARD);
	}
	if (keys['s']) {
		rayTracer.scene.cam.move(rayTracer.scene.cam.BACKWARD);
	}
	if (keys['a']) {
		rayTracer.scene.cam.move(rayTracer.scene.cam.LEFT);
	}
	if (keys['d']) {
		rayTracer.scene.cam.move(rayTracer.scene.cam.RIGHT);
	}
	if (keys['_']) {
		rayTracer.scene.cam.move(rayTracer.scene.cam.UP);
	}
	if (keys['|']) {
		rayTracer.scene.cam.move(rayTracer.scene.cam.DOWN);
	}
	if (keys['i']) {
		rayTracer.scene.lights[0].position[2] -= 1;
		rayTracer.scene.spheres[0].position[2] -= 1;
	}
	if (keys['j']) {
		rayTracer.scene.lights[0].position[0] -= 1;
		rayTracer.scene.spheres[0].position[0] -= 1;
	}
	if (keys['k']) {
		rayTracer.scene.lights[0].position[2] += 1;
		rayTracer.scene.spheres[0].position[2] += 1;
	}
	if (keys['l']) {
		rayTracer.scene.lights[0].position[0] += 1;
		rayTracer.scene.spheres[0].position[0] += 1;
	}

	// ##### fps display #####

	double currentTime = glfwGetTime();
	frameCount++;
	if (currentTime - savedTime >= 1.0) {
		rayTracer.config.fps = frameCount;
		frameCount = 0;
		savedTime = currentTime;
	}

	// ####### render #######

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	rayTracer.launchKernel();
	square.setTextureToPixels(rayTracer.framebuffer);
	square.render();

	// ######################

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

