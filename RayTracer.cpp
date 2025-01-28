#include "RayTracer.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

RayTracer::RayTracer() {

}

RayTracer::~RayTracer() {
	//delete[] objects;
	//delete[] lights;
	//free(framebuffer);
	cudaFree(framebuffer);
	cudaFree(objects);
	cudaFree(lights);
}

void RayTracer::init(int width, int height) {
	this->width = width;
	this->height = height;
	cam = Camera(vec3(0.0, 0.0, 1.0), 90.0);
	objectCount = 4;
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
	objects = new CUDASphere[objectCount];
	cudaMallocManaged((void**)&objects, objectCount * sizeof(CUDASphere));
	objects[0] = CUDASphere(vec3(0.0, 210.0, -120), 200.0f, { vec3(1, 0, 0), 0.7, 0.5, 0.0, 200.0 });
	objects[1] = CUDASphere(vec3(-40.0, 0.0, -50.0), 15.0, { vec3(0.0, 1.0, 0.0), 0.1, 0.9, 0.5, 200.0 });
	objects[2] = CUDASphere(vec3(0.0, 0.0, -50.0), 15.0, { vec3(1.0, 1.0, 0.0), 0.1, 0.9, 0.5, 200.0 });
	objects[3] = CUDASphere(vec3(40.0, 0.0, -50.0), 15.0, { vec3(0.0, 0.0, 1.0), 0.1, 0.9, 0.5, 200.0 });

	//objects[0] = CUDASphere(vec3(0.0, 0.0, -50.0), 15.0, { vec3(1.0, 1.0, 0.0), 0.3, 0.6, 0.8, 200.0 });
	//objects[1] = CUDASphere(vec3(40.0, 0.0, -50.0), 15.0, { vec3(0.0, 0.0, 1.0), 0.3, 0.6, 0.8, 200.0 });

	int lightCount = 1;
	lights = new CUDALight[lightCount];
	cudaMallocManaged((void**)&lights, lightCount * sizeof(CUDALight));
	lights[0] = { vec3(0.0, -40.0, 0.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.7, 0.0), 5, vec3(0.0, 0.0, 0.7), 2, 10 };

	cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));

	cudaMallocManaged((void**)&cam, sizeof(Camera));

	cudaDeviceSynchronize();
}

void RayTracer::launchKernel() {
	// 32x32 grid of 32x32 blocks
	// [32x32] x [32x32] = 1,048,576 threads
	// 1000 x 1000 pixels = 1,000,000 pixels on screen

	int Nx = width;
	int Ny = height;
	dim3 dimBlock(32, 32);
	dim3 dimGrid((Nx + dimBlock.x - 1) / dimBlock.x, (Ny + dimBlock.y - 1) / dimBlock.y);

	rayTrace<<<dimGrid, dimBlock>>>(width, height, framebuffer, objects, objectCount, lights, cam);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", line: " << __LINE__ << std::endl;
	}

	cudaDeviceSynchronize();
}