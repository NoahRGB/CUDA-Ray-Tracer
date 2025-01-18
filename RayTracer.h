#pragma once

#include <vector>
#include <array>

#include <gl/glew.h>
#include <glm\glm.hpp>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "Object.h"
#include "Sphere.h"
#include "utils.h"
#include "vec3.h"


class RayTracer {
private:
	int width;
	int height;
	float fov;
	float aspectRatio;

	bool inShadow(glm::vec3 pos);
	void hardShadow(glm::vec3 point, glm::vec3* colourOut);
	void softShadow(glm::vec3 point, glm::vec3* colourOut);

public:
	std::vector<Object*> objects;
	std::vector<Light> lights;
	glm::vec3 bg;

	RayTracer(int width, int height, glm::vec3 backgroundCol = glm::vec3(150.0f, 150.0f, 150.0f), float fov = 90);
	~RayTracer();

	glm::vec3 rasterToCameraSpace(float x, float y);
	glm::vec3 lighting(Material mat, glm::vec3 lightPos, glm::vec3 lightIntensity, glm::vec3 point, glm::vec3 eye, glm::vec3 normal);

	RayHit rayCast(glm::vec3 rayOrigin, glm::vec3 rayDir);
	__device__ void rayTrace(vec3* framebuffer, bool softShadows = false);
	__device__ void rayTraceTest(vec3** framebuffer);

	void addSphere(glm::vec3 pos, float radius, Material mat);
	void addLight(glm::vec3 pos, glm::vec3 colour, glm::vec3 uvec, int usteps, glm::vec3 vvec, int vsteps, int samples);
};

