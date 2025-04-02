#include "RayTracer.h"
#include "Object.h"
#include "kernels.h"
#include "Plane.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <iostream>

RayTracer::RayTracer() {

}

RayTracer::~RayTracer() {
	cudaFree(framebuffer);
	cudaFree(randStates);

	for (int i = 0; i < 2; i++) {
		cudaFree(&scenes[i].cam);

		cudaFree(scenes[i].spheres);
		cudaFree(scenes[i].planes);
		cudaFree(scenes[i].AABBs);

		for (int i = 0; i < scenes[i].modelCount; i++) {
			cudaFree(scenes[i].models[i].vertices);
			cudaFree(scenes[i].models[i].indices);
		}
		cudaFree(scenes[i].models);

		cudaFree(scenes[i].lights);
	}
}

void RayTracer::init(int width, int height) {
	this->width = width;
	this->height = height;

	dimBlock = dim3(20, 20);
	dimGrid = dim3((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	sceneCount = 3;
	scenes = new Scene[sceneCount];

	initialiseScenes();
}

void RayTracer::initialiseScenes() {

	// scene 1
	scenes[0].cam = Camera(vec3(0, 0, 30), 90.0, width / (float)height);
	scenes[0].sphereCount = 1; scenes[0].planeCount = 1; scenes[0].AABBCount = 0; scenes[0].modelCount = 1; scenes[0].lightCount = 1;

	scenes[0].spheres = new Sphere[scenes[0].sphereCount];
	cudaMallocManaged((void**)&scenes[0].spheres, scenes[0].sphereCount * sizeof(Sphere));
	scenes[0].spheres[0] = Sphere(vec3(0.0, 0.0, -50.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scenes[0].planes = new Plane[scenes[0].planeCount];
	cudaMallocManaged((void**)&scenes[0].planes, scenes[0].planeCount * sizeof(Plane));
	scenes[0].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[0].AABBs = new AABB[scenes[0].AABBCount];
	cudaMallocManaged((void**)&scenes[0].AABBs, scenes[0].AABBCount * sizeof(AABB));

	scenes[0].models = new Model[scenes[0].modelCount];
	cudaMallocManaged((void**)&scenes[0].models, scenes[0].modelCount * sizeof(Model));
	scenes[0].models[0] = Model(vec3(0.0, 20.0, 0.0), 15, "models/Chest_gold.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.0, 200.0 }, false, Reflect);

	scenes[0].lights = new Light[scenes[0].lightCount];
	cudaMallocManaged((void**)&scenes[0].lights, scenes[0].lightCount * sizeof(Light));
	scenes[0].lights[0] = { { 0.0, -40.0, -50.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[0].cam, sizeof(Camera));


	// scene 2
	scenes[1].cam = Camera(vec3(0, 0, 30), 90.0, width / (float)height);
	scenes[1].sphereCount = 2; scenes[1].planeCount = 1; scenes[1].AABBCount = 0; scenes[0].modelCount = 0; scenes[0].lightCount = 1;

	scenes[1].spheres = new Sphere[scenes[1].sphereCount];
	cudaMallocManaged((void**)&scenes[1].spheres, scenes[1].sphereCount * sizeof(Sphere));
	scenes[1].spheres[0] = Sphere(vec3(0.0, 0.0, -50.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);
	scenes[1].spheres[1] = Sphere(vec3(0.0, 0.0, 0.0), 2.0, { vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, false);

	scenes[1].planes = new Plane[scenes[1].planeCount];
	cudaMallocManaged((void**)&scenes[1].planes, scenes[1].planeCount * sizeof(Plane));
	scenes[1].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[1].AABBs = new AABB[scenes[1].AABBCount];
	cudaMallocManaged((void**)&scenes[1].AABBs, scenes[1].AABBCount * sizeof(AABB));

	scenes[1].models = new Model[scenes[1].modelCount];
	cudaMallocManaged((void**)&scenes[1].models, scenes[1].modelCount * sizeof(Model));

	scenes[1].lights = new Light[scenes[1].lightCount];
	cudaMallocManaged((void**)&scenes[1].lights, scenes[1].lightCount * sizeof(Light));
	scenes[1].lights[0] = { { 0.0, -40.0, -50.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[0].cam, sizeof(Camera));

	// scene 3
	scenes[2].cam = Camera(vec3(0, 0, 30), 90.0, width / (float)height);
	scenes[2].sphereCount = 101; scenes[2].planeCount = 1; scenes[2].AABBCount = 0; scenes[0].modelCount = 0; scenes[0].lightCount = 1;

	scenes[2].spheres = new Sphere[scenes[2].sphereCount];
	cudaMallocManaged((void**)&scenes[2].spheres, scenes[2].sphereCount * sizeof(Sphere));
	scenes[2].spheres[0] = Sphere(vec3(0.0, 0.0, -50.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	int sphereIndex = 1;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			scenes[2].spheres[sphereIndex] = Sphere(vec3(j * 15.0, 0.0, i * 15), 2.0, {vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.9, 200.0}, false);
			sphereIndex++;
		}
	}

	scenes[2].planes = new Plane[scenes[2].planeCount];
	cudaMallocManaged((void**)&scenes[2].planes, scenes[2].planeCount * sizeof(Plane));
	scenes[2].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[2].AABBs = new AABB[scenes[2].AABBCount];
	cudaMallocManaged((void**)&scenes[2].AABBs, scenes[2].AABBCount * sizeof(AABB));

	scenes[2].models = new Model[scenes[2].modelCount];
	cudaMallocManaged((void**)&scenes[2].models, scenes[2].modelCount * sizeof(Model));

	scenes[2].lights = new Light[scenes[2].lightCount];
	cudaMallocManaged((void**)&scenes[2].lights, scenes[2].lightCount * sizeof(Light));
	scenes[2].lights[0] = { { 0.0, -40.0, -50.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[0].cam, sizeof(Camera));

	////////////////////////////////////////////////////////////////////////////////////////////

	cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));

	cudaMallocManaged((void**)&randStates, dimBlock.x * dimBlock.y * dimGrid.x * dimGrid.y * sizeof(curandState));
	setupCurand<<<dimGrid, dimBlock>>>(randStates);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();

	scene = scenes[0];
}

void RayTracer::switchScene(int sceneNum) {
	scene = scenes[sceneNum];
}

void RayTracer::addSphere(vec3 pos, float radius, Material mat, ObjectType objectType) {
	scene.sphereCount++;
	Sphere* oldSpheres = scene.spheres;
	cudaFree(scene.spheres);
	cudaMallocManaged((void**)&scene.spheres, scene.sphereCount * sizeof(Sphere));

	// copy over old spheres
	for (int i = 0; i < scene.sphereCount - 1; i++) {
		scene.spheres[i] = oldSpheres[i];
	}

	// add new one on the end
	scene.spheres[scene.sphereCount - 1] = Sphere(pos, radius, mat, false, objectType);
}

void RayTracer::addPlane(vec3 pos, vec3 n, Material mat, ObjectType objectType) {
	scene.planeCount++;
	Plane* oldPlanes = scene.planes;
	cudaFree(scene.planes);
	cudaMallocManaged((void**)&scene.planes, scene.planeCount * sizeof(Plane));

	// copy over old planes
	for (int i = 0; i < scene.planeCount - 1; i++) {
		scene.planes[i] = oldPlanes[i];
	}

	// add new one on the end
	scene.planes[scene.planeCount - 1] = Plane(pos, n, mat, false, objectType);
}

void RayTracer::addAABB(vec3 pos, float size, Material mat, ObjectType objectType) {
	scene.AABBCount++;
	AABB* oldAABBs = scene.AABBs;
	cudaFree(scene.AABBs);
	cudaMallocManaged((void**)&scene.AABBs, scene.AABBCount * sizeof(AABB));

	// copy over old boxes
	for (int i = 0; i < scene.AABBCount - 1; i++) {
		scene.AABBs[i] = oldAABBs[i];
	}

	// add new one on the end
	scene.AABBs[scene.AABBCount - 1] = AABB(pos, size, mat, false, objectType);
}

void RayTracer::launchKernel() {
	// 32x32 grid of 32x32 blocks
	// [32x32] x [32x32] = 1,048,576 threads
	// 1000 x 1000 pixels = 1,000,000 pixels on screen

	rayTrace<<<dimGrid, dimBlock>>>(width, height, framebuffer, scene, config, randStates);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();
}