#pragma once

#include "vec3.h"
#include "Camera.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"
#include "Model.h"
#include "Box.h"

#include <gl/glew.h>
#include <map>

#include <curand_kernel.h>
#include <random>

__host__ __device__ struct Scene {
	Sphere* spheres;
	int sphereCount;

	Plane* planes;
	int planeCount;

	Box* boxes;
	int boxCount;

	Triangle* triangles;
	int triangleCount;

	Light* lights;
	int lightCount = 1;

	Model* models;
	int modelCount = 1;

	Camera cam;
};

__host__ __device__ struct SceneConfig {
	int fps;
	float shadowBias = 0.01;
	bool renderHardShadows = false;
	bool renderSoftShadows = false;
	bool areaLightSpecularEffect = false;
	bool reflections = false;
	int maxDepth = 2;
	int softShadowRadius = 5;
	int softShadowNum = 15;
	float sphereReflectionStrength = 0.5;
	float planeReflectionStrength = 0.5;
	float boxReflectionStrength = 0.5;
	float shadowIntensity = 0.5;
	bool renderBoxes = false;
	bool ambientLighting = true;
	bool diffuseLighting = true;
	bool specularLighting = true;
	bool antiAliasing = false;
	vec3 backgroundCol = vec3(0.8, 0.8, 0.8);
	int backgroundBrightness = 2;
	int floorBrightness = 4;
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

