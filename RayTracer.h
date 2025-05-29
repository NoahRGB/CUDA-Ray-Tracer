#pragma once

#include "vec3.h"
#include "Camera.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"
#include "Model.h"
#include "AABB.h"

#include <gl/glew.h>
#include <map>

#include <curand_kernel.h>
#include <random>

__host__ __device__ struct Scene {
	Sphere* spheres;
	int sphereCount;

	Plane* planes;
	int planeCount;

	AABB* AABBs;
	int AABBCount;

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
	int softShadowRadius = 5;
	int softShadowNum = 15;
	float shadowIntensity = 0.5;

	bool areaLightSpecularEffect = false;
	bool reflections = false;
	int maxDepth = 2;
	bool reflectPlanes = false;
	float sphereReflectionStrength = 0.5;
	float planeReflectionStrength = 0.5;
	float AABBReflectionStrength = 0.5;
	float modelReflectionStrength = 0.5;

	bool renderAABBs = true;
	bool renderModels = false;

	bool ambientLighting = true;
	bool diffuseLighting = true;
	bool specularLighting = true;
	bool antiAliasing = false;
	bool boundingBox = true;

	vec3 backgroundCol = vec3(0.1, 0.1, 0.1);
	int backgroundBrightness = 1;
	int floorBrightness = 10;
};

class RayTracer {
public:
	Scene scene;
	SceneConfig config;

	Scene* scenes;
	int sceneCount;

	dim3 dimBlock;
	dim3 dimGrid;

	GLubyte* framebuffer;

	curandState* randStates;

	int width, height;

	RayTracer();
	~RayTracer();

	void init(int width, int height);

	void initialiseScenes();
	void switchScene(int sceneNum);
	void launchKernel();

	void addSphere(vec3 pos, float radius, Material mat, ObjectType objectType = Diffuse);
	void addPlane(vec3 pos, vec3 n, Material mat, ObjectType objectType = Diffuse);
	void addAABB(vec3 pos, float size, Material mat, ObjectType objectType = Diffuse);
};

