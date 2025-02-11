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
	float shadowBias = 0.01;
	bool renderHardShadows = false;
	bool reflections = true;
	int maxDepth = 2;
	float reflectionStrength = 0.5;
	float shadowIntensity = 0.0;
	bool ambientLighting = true;
	bool diffuseLighting = true;
	bool specularLighting = true;
	bool antiAliasing = false;
	vec3 backgroundCol = vec3(0.8, 0.8, 0.8);
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
	void addSphere(vec3 pos, float radius, CUDAMaterial mat, ObjectType objectType = Diffuse);
};

