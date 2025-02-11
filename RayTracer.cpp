#include "RayTracer.h"
#include "Object.h"
#include "kernels.h"
#include "Plane.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

RayTracer::RayTracer() {

}

RayTracer::~RayTracer() {
	cudaFree(framebuffer);
	cudaFree(scene.spheres);
	cudaFree(scene.planes);
	cudaFree(scene.lights);
	cudaFree(&scene.cam);
}

void RayTracer::init(int width, int height) {
	this->width = width;
	this->height = height;

	scene.cam = Camera(vec3(0, 0, -0), 90.0, width / (float)height, 0.0, -90.0);
	scene.sphereCount = 1;
	scene.planeCount = 1;
	scene.lightCount = 1;
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
	scene.spheres = new CUDASphere[scene.sphereCount];
	cudaMallocManaged((void**)&scene.spheres, scene.sphereCount * sizeof(CUDASphere));
	//spheres[0] = CUDASphere(vec3(0.0, 210.0, -120), 200.0f, { vec3(1, 0, 0), 0.7, 0.5, 0.0, 200.0 });
	//scene.spheres[0] = CUDASphere(vec3(-40.0, 0.0, -50.0), 15.0, { vec3(0.0, 1.0, 0.0), 0.1, 0.9, 0.5, 200.0 });
	//scene.spheres[1] = CUDASphere(vec3(0.0, 0.0, -50.0), 15.0, { vec3(1.0, 1.0, 0.0), 0.1, 0.9, 0.5, 200.0 });
	//scene.spheres[2] = CUDASphere(vec3(40.0, 0.0, -50.0), 15.0, { vec3(0.0, 0.0, 1.0), 0.1, 0.9, 0.5, 200.0 });
	scene.spheres[0] = CUDASphere(vec3(0.0, -40.0, -50.0), 2.0, { vec3(0.0, 1.0, 0.0), 1.0, 0.0, 0.0, 200.0 }, true);
	//scene.spheres[1] = CUDASphere(vec3(0.0, 0.0, 0.0), 20.0, { vec3(1.0, 0.0, 0.0), 0.1, 0.5, 0.5, 200.0 });
	//scene.spheres[2] = CUDASphere(vec3(0.0, 0.0, 50.0), 20.0, { vec3(1.0, 0.0, 0.0), 0.1, 0.5, 0.5, 200.0 });

	scene.planes = new Plane[scene.planeCount];
	cudaMallocManaged((void**)&scene.planes, scene.planeCount * sizeof(Plane));
	scene.planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.4, 0.4, 0.4), 0.0, 0.5, 0.0, 200 }, false, Reflect);

	scene.lights = new CUDALight[scene.lightCount];
	cudaMallocManaged((void**)&scene.lights, scene.lightCount * sizeof(CUDALight));
	scene.lights[0] = { vec3(0.0, -40.0, -50.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), 5, vec3(1.0, 0.0, 0.0), 2, 10 };

	cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));

	cudaMallocManaged((void**)&scene.cam, sizeof(Camera));

	cudaDeviceSynchronize();
}

void RayTracer::addSphere(vec3 pos, float radius, CUDAMaterial mat, ObjectType objectType) {
	scene.sphereCount++;
	CUDASphere* oldSpheres = scene.spheres;
	cudaFree(scene.spheres);
	cudaMallocManaged((void**)&scene.spheres, scene.sphereCount * sizeof(CUDASphere));

	// copy over old spheres
	for (int i = 0; i < scene.sphereCount - 1; i++) {
		scene.spheres[i] = oldSpheres[i];
	}

	// add new one on the end
	scene.spheres[scene.sphereCount - 1] = CUDASphere(pos, radius, mat, false, objectType);
}

void RayTracer::launchKernel() {
	// 32x32 grid of 32x32 blocks
	// [32x32] x [32x32] = 1,048,576 threads
	// 1000 x 1000 pixels = 1,000,000 pixels on screen

	int Nx = width;
	int Ny = height;
	dim3 dimBlock(16, 16);
	dim3 dimGrid((Nx + dimBlock.x - 1) / dimBlock.x, (Ny + dimBlock.y - 1) / dimBlock.y);

	rayTrace<<<dimGrid, dimBlock>>>(width, height, framebuffer, scene, config);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();
}