#pragma once

#include "CUDASphere.h"
#include "Camera.h"
#include "Plane.h"
#include "vec3.h"

#include <gl/glew.h>
#include <map>

__host__ __device__ struct Scene {
	CUDASphere* spheres;
	int sphereCount;

	Plane* planes;
	int planeCount;

	CUDALight* lights;
	int lightCount = 1;

	Camera cam;
};

__host__ __device__ struct SceneConfig {
	int fps;
	float shadowBias = 0.5;
	bool renderHardShadows = true;
	bool reflections = true;
	bool ambientLighting = true;
	bool diffuseLighting = true;
	bool specularLighting = true;
};

class RayTracer {
public:
	Scene scene;
	SceneConfig config;

	GLubyte* framebuffer;

	int width, height;

	RayTracer();
	~RayTracer();

	void init(int width, int height);

	void initialiseScene();
	void launchKernel();
	void resize(int width, int height);
	void addSphere(vec3 pos, float radius, CUDAMaterial mat);
};

