#pragma once

#include "vec3.h"
#include "Camera.h"
#include "Sphere.h"
#include "Plane.h"
#include "Box.h"

#include <gl/glew.h>
#include <map>

#include <curand_kernel.h>

__host__ __device__ struct Scene {
	Sphere* spheres;
	int sphereCount;

	Plane* planes;
	int planeCount;

	Box* boxes;
	int boxCount;

	Light* lights;
	int lightCount = 1;

	Camera cam;
};

__host__ __device__ struct SceneConfig {
	int fps;
	float shadowBias = 0.01;
	bool renderHardShadows = false;
	bool renderSoftShadows = false;
	bool reflections = true;
	int maxDepth = 2;
	int softShadowNum = 10;
	int softShadowRadius = 10;
	int dampning = 15;
	float sphereReflectionStrength = 0.5;
	float planeReflectionStrength = 0.5;
	float boxReflectionStrength = 0.5;
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

	dim3 dimBlock;
	dim3 dimGrid;

	GLubyte* framebuffer;

	curandState* randStates;

	int width, height;

	RayTracer();
	~RayTracer();

	void init(int width, int height);

	void initialiseScene();
	void launchKernel();

	void resize(int width, int height);

	void addSphere(vec3 pos, float radius, Material mat, ObjectType objectType = Diffuse);
	void addPlane(vec3 pos, vec3 n, Material mat, ObjectType objectType = Diffuse);
	void addBox(vec3 pos, float size, Material mat, ObjectType objectType = Diffuse);
};

