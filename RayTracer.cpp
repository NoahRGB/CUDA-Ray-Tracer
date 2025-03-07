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
	cudaFree(&scene.cam);

	cudaFree(scene.spheres);
	cudaFree(scene.planes);
	cudaFree(scene.boxes);
	cudaFree(scene.lights);
}

void RayTracer::init(int width, int height) {
	this->width = width;
	this->height = height;

	scene.cam = Camera(vec3(0, 0, -0), 90.0, width / (float)height, 0.0, -90.0);
	scene.sphereCount = 1;
	scene.planeCount = 1;
	scene.boxCount = 0;
	scene.lightCount = 1;

	dimBlock = dim3(20, 20);
	dimGrid = dim3((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	initialiseScene();
}

void RayTracer::resize(int width, int height) {

	this->width = width;
	this->height = height;

	cudaError_t err;
	err = cudaFree(framebuffer);
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	err = cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();
}

void RayTracer::initialiseScene() {
	scene.spheres = new Sphere[scene.sphereCount];
	cudaMallocManaged((void**)&scene.spheres, scene.sphereCount * sizeof(Sphere));
	scene.spheres[0] = Sphere(vec3(0.0, -40.0, -50.0), 2.0, { vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scene.planes = new Plane[scene.planeCount];
	cudaMallocManaged((void**)&scene.planes, scene.planeCount * sizeof(Plane));
	scene.planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), 0.9, 0.5, 0.0, 200 }, false, Reflect);

	scene.boxes = new Box[scene.boxCount];
	cudaMallocManaged((void**)&scene.boxes, scene.boxCount * sizeof(Box));
	//scene.boxes[0] = Box(vec3(0.0, 0.0, 0.0), 10.0, { vec3(0.0, 1.0, 0.0), 0.1, 0.8, 0.0, 200 }, false, Diffuse);

	scene.lights = new Light[scene.lightCount];
	cudaMallocManaged((void**)&scene.lights, scene.lightCount * sizeof(Light));
	scene.lights[0] = { { 0.0, -40.0, -50.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };


	cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));

	cudaMallocManaged((void**)&scene.cam, sizeof(Camera));

	cudaMallocManaged((void**)&randStates, dimBlock.x * dimBlock.y * dimGrid.x * dimGrid.y * sizeof(curandState));

	setupCurand<<<dimGrid, dimBlock>>>(randStates);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();
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

void RayTracer::addBox(vec3 pos, float size, Material mat, ObjectType objectType) {
	scene.boxCount++;
	Box* oldBoxes = scene.boxes;
	cudaFree(scene.boxes);
	cudaMallocManaged((void**)&scene.boxes, scene.boxCount * sizeof(Box));

	// copy over old boxes
	for (int i = 0; i < scene.boxCount - 1; i++) {
		scene.boxes[i] = oldBoxes[i];
	}

	// add new one on the end
	scene.boxes[scene.boxCount - 1] = Box(pos, size, mat, false, objectType);
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