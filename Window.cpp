#include "Window.h"
#include "RayTracer.h"
#include "vec3.h"
#include "kernels.h"
#include "Sphere.h"
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
	lastRenderTime = 0;
	measuringRenderTimes = true;
	averageRenderTime = 0;
	totalRenderTimes = 0;
	numRendersMeasured = 0;
	maxRendersToMeasure = 100;

	mouseEnabled = true;
	firstMouse = true;
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
		if (key == GLFW_KEY_U) win->keys['u'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['u'];
		if (key == GLFW_KEY_O) win->keys['o'] = action == GLFW_PRESS ? true : action == GLFW_RELEASE ? false : win->keys['o'];

		if (key == GLFW_KEY_H) {
			std::cout << win->rayTracer.scene.cam.getPosition().x() << ", " << win->rayTracer.scene.cam.getPosition().y() << ", " << win->rayTracer.scene.cam.getPosition().z() << " and " << win->rayTracer.scene.cam.yaw << ", " << win->rayTracer.scene.cam.pitch << std::endl;
			std::cout << "Light: " << win->rayTracer.scene.lights[0].position.x() << ", " << win->rayTracer.scene.lights[0].position.y() << ", " << win->rayTracer.scene.lights[0].position.z() << std::endl;
		}

		if (key == GLFW_KEY_R && action == GLFW_PRESS) {
			win->startRenderTimeMeasure();
		}

		if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_PRESS) {
			if (!win->mouseEnabled) {
				win->mouseEnabled = true;
				win->firstMouse = true;
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else {
				win->mouseEnabled = false;
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}

		if (key == GLFW_KEY_Z && action == GLFW_PRESS) { // empty scene
			win->rayTracer.switchScene(0);
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_X && action == GLFW_PRESS) { // spheres for reflections
			win->rayTracer.switchScene(1);
			win->rayTracer.config.maxDepth = 1;
			win->rayTracer.config.renderHardShadows = false;
			win->rayTracer.config.renderSoftShadows = false;
			win->rayTracer.config.reflections = true;
			//win->rayTracer.config.backgroundBrightness = 7;
			//win->rayTracer.config.floorBrightness = 3;
			win->rayTracer.config.reflectPlanes = true;
			win->rayTracer.config.planeReflectionStrength = 0.5;
			win->rayTracer.config.sphereReflectionStrength = 0.7;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_C && action == GLFW_PRESS) { // spheres for shadows
			win->rayTracer.switchScene(5);
			win->rayTracer.config.reflections = false;
			win->rayTracer.config.renderHardShadows = true;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_V && action == GLFW_PRESS) { // 100 spheres
			win->rayTracer.switchScene(2);
			win->rayTracer.config.renderHardShadows = false;
			win->rayTracer.config.renderSoftShadows = false;
			win->rayTracer.config.reflections = false;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_B && action == GLFW_PRESS) { // boxes
			win->rayTracer.switchScene(3);
			win->rayTracer.config.reflections = true;
			win->rayTracer.config.planeReflectionStrength = 0.22;
			win->rayTracer.config.renderHardShadows = false;
			win->rayTracer.config.renderSoftShadows = false;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_N && action == GLFW_PRESS) { // reflective boxes + spheres
			win->rayTracer.switchScene(7);
			win->rayTracer.config.renderHardShadows = false;
			win->rayTracer.config.renderSoftShadows = false;
			win->rayTracer.config.reflections = true;
			win->rayTracer.config.reflectPlanes = true;
			win->rayTracer.config.maxDepth = 1;
			win->rayTracer.config.planeReflectionStrength = 0.5;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_M && action == GLFW_PRESS) { // single chest model
			win->rayTracer.config.renderHardShadows = false;
			win->rayTracer.config.renderSoftShadows = false;
			win->rayTracer.config.reflections = false;
			win->rayTracer.switchScene(6);
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_COMMA && action == GLFW_PRESS) { // reflective box + model
			win->rayTracer.switchScene(8);
			win->rayTracer.config.reflections = true;
			win->rayTracer.config.reflectPlanes = true;
			win->rayTracer.config.planeReflectionStrength = 0.5;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
		}
		if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS) { // table + book + cup close up
			win->rayTracer.switchScene(4);
			win->rayTracer.config.renderHardShadows = true;
			win->rayTracer.config.renderSoftShadows = false;
			win->rayTracer.config.reflections = true;
			win->rayTracer.config.modelReflectionStrength = 0.8;
			win->rayTracer.scene.cam.mouseMovement(0, 0);
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
	savedFrameTime = glfwGetTime();
	frameCount = 0;
	while (!glfwWindowShouldClose(window) && !glfwGetKey(window, GLFW_KEY_ESCAPE)) {
		glfwSwapBuffers(window);
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Ray Tracing");
		ImGui::Text("FPS: %d", rayTracer.config.fps);
		ImGui::Text("Render time: %f", lastRenderTime);
		ImGui::Text("Average render time: %f over %d renders", averageRenderTime, maxRendersToMeasure);
		ImGui::Checkbox("Anti aliasing?", &rayTracer.config.antiAliasing);
		ImGui::Checkbox("Use bounding boxes?", &rayTracer.config.boundingBox);
		ImGui::Checkbox("Use 8 bounding boxes?", &rayTracer.config.eightBoundingBoxes);
		ImGui::Checkbox("Cull back triangles?", &rayTracer.config.cullBackTriangles);
		ImGui::Checkbox("Render boxes", &rayTracer.config.renderAABBs);
		ImGui::Checkbox("Render models", &rayTracer.config.renderModels);
		ImGui::SliderInt("Background brightness", &rayTracer.config.backgroundBrightness, 1, 10);
		ImGui::SliderInt("Floor brightness", &rayTracer.config.floorBrightness, 1, 10);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::CollapsingHeader("Lighting options")) {
			ImGui::SeparatorText("Lighting");
			ImGui::Checkbox("Hard Shadows", &rayTracer.config.renderHardShadows);
			ImGui::SliderFloat("Hard shadow intensity", &rayTracer.config.shadowIntensity, 0.0, 1.0);
			ImGui::Checkbox("Soft Shadows", &rayTracer.config.renderSoftShadows);
			ImGui::SliderInt("Soft shadow radius", &rayTracer.config.softShadowRadius, 1, 15);
			ImGui::SliderInt("Soft shadow casts", &rayTracer.config.softShadowNum, 1, 50);
			ImGui::SliderFloat("Shadow bias", &rayTracer.config.shadowBias, 0.0, 15.0);
			ImGui::Checkbox("Area Light Specular", &rayTracer.config.areaLightSpecularEffect);
			ImGui::Checkbox("Ambient lighting", &rayTracer.config.ambientLighting);
			ImGui::Checkbox("Diffuse lighting", &rayTracer.config.diffuseLighting);
			ImGui::Checkbox("Specular lighting", &rayTracer.config.specularLighting);
			ImGui::Checkbox("Reflections", &rayTracer.config.reflections);
			ImGui::SliderInt("Reflection depth", &rayTracer.config.maxDepth, 1, 2);
			ImGui::Checkbox("Reflect planes", &rayTracer.config.reflectPlanes);
			ImGui::SliderFloat("Sphere Reflection strength", &rayTracer.config.sphereReflectionStrength, 0.0, 1.0);
			ImGui::SliderFloat("Plane Reflection strength", &rayTracer.config.planeReflectionStrength, 0.0, 1.0);
			ImGui::SliderFloat("Box Reflection strength", &rayTracer.config.AABBReflectionStrength, 0.0, 1.0);
			ImGui::SliderFloat("Model Reflection strength", &rayTracer.config.modelReflectionStrength, 0.0, 1.0);
		}
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::CollapsingHeader("Create sphere")) {
			ImGui::SeparatorText("Create sphere");
			static vec3 pos, ambientCol(1.0, 0.0, 0.0), diffuseCol(1.0, 0.0, 0.0);
			static float radius = 20.0;
			static float ambient = 0.1, diffuse = 0.9, specular = 0.5, shininess = 200.0;
			static bool reflective = false;
			ImGui::InputFloat3("Pos", pos.nums);
			ImGui::InputFloat("Radius", &radius);
			ImGui::InputFloat3("Ambient colour", ambientCol.nums);
			ImGui::InputFloat3("Diffuse colour", diffuseCol.nums);
			ImGui::InputFloat("Ambient", &ambient);
			ImGui::InputFloat("Diffuse", &diffuse);
			ImGui::InputFloat("Specular", &specular);
			ImGui::InputFloat("Shininess", &shininess);
			ImGui::Checkbox("Reflective?", &reflective);
			if (ImGui::Button("Create Sphere")) {
				rayTracer.addSphere(pos, radius, { ambientCol, diffuseCol, ambient, diffuse, specular, shininess }, reflective ? Reflect : Diffuse);
			}
		}
		//ImGui::Dummy(ImVec2(0.0f, 20.0f));
		//if (ImGui::CollapsingHeader("Create plane")) {
		//	ImGui::SeparatorText("Create plane");
		//	static vec3 pos, normal, ambientCol(1.0, 0.0, 0.0), diffuseCol(1.0, 0.0, 0.0);
		//	static float ambient = 0.1, diffuse = 0.9, specular = 0.5, shininess = 200.0;
		//	static bool reflective = false;
		//	ImGui::InputFloat3("Pos", pos.nums);
		//	ImGui::InputFloat3("Ambient colour", ambientCol.nums);
		//	ImGui::InputFloat3("Diffuse colour", diffuseCol.nums);
		//	ImGui::InputFloat3("Direction", normal.nums);
		//	ImGui::InputFloat("Ambient", &ambient);
		//	ImGui::InputFloat("Diffuse", &diffuse);
		//	ImGui::InputFloat("Specular", &specular);
		//	ImGui::InputFloat("Shininess", &shininess);
		//	ImGui::Checkbox("Reflective?", &reflective);
		//	if (ImGui::Button("Create Plane")) {
		//		rayTracer.addPlane(pos, normal, { ambientCol, diffuseCol, ambient, diffuse, specular, shininess}, reflective ? Reflect : Diffuse);
		//	}
		//}
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::CollapsingHeader("Create box")) {
			ImGui::SeparatorText("Create AABB");
			static vec3 pos, ambientCol(1.0, 0.0, 0.0), diffuseCol(1.0, 0.0, 0.0);
			static float size = 10.0, ambient = 0.1, diffuse = 0.9, specular = 0.5, shininess = 200.0;
			static bool reflective = false;
			ImGui::InputFloat3("Pos", pos.nums);
			ImGui::InputFloat("Size", &size);
			ImGui::InputFloat3("Ambient colour", ambientCol.nums);
			ImGui::InputFloat3("Diffuse colour", diffuseCol.nums);
			ImGui::InputFloat("Ambient", &ambient);
			ImGui::InputFloat("Diffuse", &diffuse);
			ImGui::InputFloat("Specular", &specular);
			ImGui::InputFloat("Shininess", &shininess);
			ImGui::Checkbox("Reflective?", &reflective);
			if (ImGui::Button("Create AABB")) {
				rayTracer.addAABB(pos, size, { ambientCol, diffuseCol, ambient, diffuse, specular, shininess }, reflective ? Reflect : Diffuse);
			}
		}

		ImGui::End();

		display();
	}
	glfwTerminate();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Window::startRenderTimeMeasure() {
	numRendersMeasured = 0;
	totalRenderTimes = 0;
}

void Window::stopRenderTimeMeasure() {
	averageRenderTime = totalRenderTimes / numRendersMeasured;
	startRenderTimeMeasure();
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
	if (keys['u']) {
		rayTracer.scene.lights[0].position[1] += 1;
		rayTracer.scene.spheres[0].position[1] += 1;
	}
	if (keys['o']) {
		rayTracer.scene.lights[0].position[1] -= 1;
		rayTracer.scene.spheres[0].position[1] -= 1;
	}

	rayTracer.scene.spheres[0].radius = rayTracer.config.softShadowRadius;
	rayTracer.scene.planes[0].mat.ambientColour = vec3(0.1, 0.1, 0.1) * (float)rayTracer.config.floorBrightness;
	rayTracer.config.backgroundCol = vec3(0.1, 0.1, 0.1) * (float)rayTracer.config.backgroundBrightness;

	// ##### fps display #####

	double currentTime = glfwGetTime();
	frameCount++;
	if (currentTime - savedFrameTime >= 1.0) {
		rayTracer.config.fps = frameCount;
		frameCount = 0;
		savedFrameTime = currentTime;
	}

	// ####### render #######

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	double timeBefore = glfwGetTime();
	rayTracer.launchKernel();
	double timeAfter = glfwGetTime();
	lastRenderTime = timeAfter - timeBefore;


	if (measuringRenderTimes) {
		numRendersMeasured++;
		totalRenderTimes += lastRenderTime;
		if (numRendersMeasured == maxRendersToMeasure) {
			stopRenderTimeMeasure();
		}
	}

	square.setTextureToPixels(rayTracer.framebuffer);
	square.render();

	// ######################

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

