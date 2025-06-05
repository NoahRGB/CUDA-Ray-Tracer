#include "RayTracer.h"
#include "Object.h"
#include "kernels.h"
#include "Plane.h"
#include "utils.h"
#include "queue.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <iostream>

RayTracer::RayTracer() {

}

RayTracer::~RayTracer() {
	cudaFree(framebuffer);
	cudaFree(randStates);

	for (int i = 0; i < 3; i++) {
		cudaFree(&scenes[i].cam);

		cudaFree(scenes[i].spheres);
		cudaFree(scenes[i].planes);
		cudaFree(scenes[i].AABBs);

		for (int j = 0; j < scenes[i].modelCount; j++) {
			cudaFree(scenes[i].models[j].vertices);
			cudaFree(scenes[i].models[j].indices);
			for (int k = 0; k < 8; k++) {
				cudaFree(scenes[i].models[j].boundingBoxes[k].includedModelIndices);
			}

			//Queue queue;
			//queue.enqueue(*scenes[i].models[j].octree.root);
			//while (queue.used != 0) {
			//	OctreeNode node = queue.dequeue();
			//	cudaFree(&node.includedVertices);
			//	if (!node.isLeaf) {
			//		for (int k = 0; k < 8; k++) {
			//			queue.enqueue(node.children[k]);
			//		}
			//		cudaFree(&node.children);
			//	}
			//}



		}
		cudaFree(scenes[i].models);

		cudaFree(scenes[i].lights);
	}
}

void RayTracer::init(int width, int height) {
	this->width = width;
	this->height = height;

	dimBlock = dim3(16, 16);
	dimGrid = dim3((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	sceneCount = 9;
	scenes = new Scene[sceneCount];

	initialiseScenes();
}

void RayTracer::initialiseScenes() {

	// scene 1 - empty
	scenes[0].cam = Camera(vec3(0.0, 0.0, 50.0), 90.0, width / (float)height);
	scenes[0].sphereCount = 1; scenes[0].planeCount = 1; scenes[0].AABBCount = 0; scenes[0].modelCount = 0; scenes[0].lightCount = 1;

	scenes[0].spheres = new Sphere[scenes[0].sphereCount];
	cudaMallocManaged((void**)&scenes[0].spheres, scenes[0].sphereCount * sizeof(Sphere));
	scenes[0].spheres[0] = Sphere(vec3(0.0, 0.0, -50.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scenes[0].planes = new Plane[scenes[0].planeCount];
	cudaMallocManaged((void**)&scenes[0].planes, scenes[0].planeCount * sizeof(Plane));
	scenes[0].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[0].models = new Model[scenes[0].modelCount];
	cudaMallocManaged((void**)&scenes[0].models, scenes[0].modelCount * sizeof(Model));

	scenes[0].AABBs = new AABB[scenes[0].AABBCount];
	cudaMallocManaged((void**)&scenes[0].AABBs, scenes[0].AABBCount * sizeof(AABB));

	scenes[0].lights = new Light[scenes[0].lightCount];
	cudaMallocManaged((void**)&scenes[0].lights, scenes[0].lightCount * sizeof(Light));
	scenes[0].lights[0] = { { 0.0, -40.0, -50.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[0].cam, sizeof(Camera));


	// scene 2 - spheres for reflections
	scenes[1].cam = Camera(vec3(0, 0, 0), 90.0, width / (float)height);
	scenes[1].sphereCount = 4; scenes[1].planeCount = 1; scenes[1].AABBCount = 1; scenes[1].modelCount = 0; scenes[1].lightCount = 1;

	scenes[1].spheres = new Sphere[scenes[1].sphereCount];
	cudaMallocManaged((void**)&scenes[1].spheres, scenes[1].sphereCount * sizeof(Sphere));
	scenes[1].spheres[0] = Sphere(vec3(-30.0, -70.0, 0.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);
	scenes[1].spheres[1] = Sphere(vec3(0.0, 0.0, -70.0), 15.0, { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.2, 0.2, 0.0, 200.0 }, false, Reflect);
	scenes[1].spheres[2] = Sphere(vec3(50.0, 0.0, -70.0), 15.0, { vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), 0.2, 0.2, 0.0, 200.0 }, false, Reflect);
	scenes[1].spheres[3] = Sphere(vec3(0.0, -100.0, -240.0), 100.0, { vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), 0.2, 0.2, 0.0, 200.0 }, false, Reflect);

	scenes[1].planes = new Plane[scenes[1].planeCount];
	cudaMallocManaged((void**)&scenes[1].planes, scenes[1].planeCount * sizeof(Plane));
	scenes[1].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[1].AABBs = new AABB[scenes[1].AABBCount];
	cudaMallocManaged((void**)&scenes[1].AABBs, scenes[1].AABBCount * sizeof(AABB));
	scenes[1].AABBs[0] = AABB(vec3(-50.0, 0.0, -70.0), 15.0, { vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), 0.2, 0.2, 0.0, 200.0 }, false, Reflect);

	scenes[1].models = new Model[scenes[1].modelCount];
	cudaMallocManaged((void**)&scenes[1].models, scenes[1].modelCount * sizeof(Model));

	scenes[1].lights = new Light[scenes[1].lightCount];
	cudaMallocManaged((void**)&scenes[1].lights, scenes[1].lightCount * sizeof(Light));
	scenes[1].lights[0] = { { -30.0, -70.0, 0.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[1].cam, sizeof(Camera));


	// scene 3 - 100 spheres
	scenes[2].cam = Camera(vec3(-50, 0, 0), 90.0, width / (float)height);
	scenes[2].sphereCount = 101; scenes[2].planeCount = 1; scenes[2].AABBCount = 0; scenes[2].modelCount = 0; scenes[2].lightCount = 1;

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

	cudaMallocManaged((void**)&scenes[2].cam, sizeof(Camera));


	// scene 4 - boxes
	scenes[3].cam = Camera(vec3(1.75159, -55.2005, 144.58), 90.0, width / (float)height, -93.5, 6.6);
	scenes[3].sphereCount = 1; scenes[3].planeCount = 1; scenes[3].AABBCount = 12; scenes[3].modelCount = 0; scenes[3].lightCount = 1;

	scenes[3].spheres = new Sphere[scenes[3].sphereCount];
	cudaMallocManaged((void**)&scenes[3].spheres, scenes[3].sphereCount * sizeof(Sphere));
	scenes[3].spheres[0] = Sphere(vec3(-30.0, -70.0, 0.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scenes[3].planes = new Plane[scenes[3].planeCount];
	cudaMallocManaged((void**)&scenes[3].planes, scenes[3].planeCount * sizeof(Plane));
	scenes[3].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[3].AABBs = new AABB[scenes[3].AABBCount];
	cudaMallocManaged((void**)&scenes[3].AABBs, scenes[3].AABBCount * sizeof(AABB));

	scenes[3].AABBs[0] = AABB(vec3(100.0, 0.0, 0.0), 100.0, { vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);
	scenes[3].AABBs[3] = AABB(vec3(100.0, -100.0, 0.0), 100.0, { vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);
	scenes[3].AABBs[8] = AABB(vec3(100.0, 0.0, 100.0), 100.0, { vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);
	scenes[3].AABBs[9] = AABB(vec3(100.0, -100.0, 100.0), 100.0, { vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);

	scenes[3].AABBs[1] = AABB(vec3(-100.0, 0.0, 0.0), 100.0, { vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);
	scenes[3].AABBs[4] = AABB(vec3(-100.0, -100.0, 0.0), 100.0, { vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);
	scenes[3].AABBs[10] = AABB(vec3(-100.0, 0.0, 100.0), 100.0, { vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);
	scenes[3].AABBs[11] = AABB(vec3(-100.0, -100.0, 100.0), 100.0, { vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), 0.2, 0.2, 0.3, 10 }, false, Reflect);

	scenes[3].AABBs[2] = AABB(vec3(0.0, 0.0, -100.0), 100.0, { vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0), 0.2, 0.2, 0.3, 10 }, false, Diffuse);
	scenes[3].AABBs[5] = AABB(vec3(0.0, -100.0, -100.0), 100.0, { vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0), 0.2, 0.2, 0.3, 10 }, false, Diffuse);

	scenes[3].AABBs[6] = AABB(vec3(0.0, 0.0, 10.0), 40.0, { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.2, 0.8, 0.9, 200 }, false, Diffuse);
	scenes[3].AABBs[7] = AABB(vec3(0.0, -30.0, 10.0), 20.0, { vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), 0.2, 0.8, 0.9, 200 }, false, Diffuse);

	scenes[3].models = new Model[scenes[3].modelCount];
	cudaMallocManaged((void**)&scenes[3].models, scenes[3].modelCount * sizeof(Model));

	scenes[3].lights = new Light[scenes[3].lightCount];
	cudaMallocManaged((void**)&scenes[3].lights, scenes[3].lightCount * sizeof(Light));
	scenes[3].lights[0] = { { -30.0, -70.0, 0.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[3].cam, sizeof(Camera));


	// scene 5 - table + book + cup
	scenes[4].cam = Camera(vec3(-11.2339, -4.81805, -13.431), 90.0, width / (float)height, -311.2, 13.8);
	scenes[4].sphereCount = 1; scenes[4].planeCount = 1; scenes[4].AABBCount = 0; scenes[4].modelCount = 3; scenes[4].lightCount = 1;

	scenes[4].spheres = new Sphere[scenes[4].sphereCount];
	cudaMallocManaged((void**)&scenes[4].spheres, scenes[4].sphereCount * sizeof(Sphere));
	scenes[4].spheres[0] = Sphere(vec3(-30.0, -70.0, 0.0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scenes[4].planes = new Plane[scenes[4].planeCount];
	cudaMallocManaged((void**)&scenes[4].planes, scenes[4].planeCount * sizeof(Plane));
	scenes[4].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[4].AABBs = new AABB[scenes[4].AABBCount];
	cudaMallocManaged((void**)&scenes[4].AABBs, scenes[4].AABBCount * sizeof(AABB));

	scenes[4].models = new Model[scenes[4].modelCount];
	cudaMallocManaged((void**)&scenes[4].models, scenes[4].modelCount * sizeof(Model));
	scenes[4].models[0] = Model(vec3(0.0, 19.0, 0.0), 15, "models/Table.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.0, 10.0 }, false, Diffuse);
	scenes[4].models[1] = Model(vec3(0.0, 6.0, 10.0), 10, "models/Chalice.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 1.0, 1000.0 }, false, Reflect);
	scenes[4].models[2] = Model(vec3(0.0, 6.0, 0.0), 10, "models/Book2_Open.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.0, 10.0 }, false, Diffuse);

	scenes[4].lights = new Light[scenes[4].lightCount];
	cudaMallocManaged((void**)&scenes[4].lights, scenes[4].lightCount * sizeof(Light));
	scenes[4].lights[0] = { { -30.0, -70.0, 0.0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[4].cam, sizeof(Camera));


	// scene 6 - spheres for shadows
	scenes[5].cam = Camera(vec3(0, 0, 0), 90.0, width / (float)height);
	scenes[5].sphereCount = 4; scenes[5].planeCount = 1; scenes[5].AABBCount = 0; scenes[5].modelCount = 0; scenes[5].lightCount = 1;

	scenes[5].spheres = new Sphere[scenes[5].sphereCount];
	cudaMallocManaged((void**)&scenes[5].spheres, scenes[5].sphereCount * sizeof(Sphere));
	scenes[5].spheres[0] = Sphere(vec3(-29, -32, 0), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);
	scenes[5].spheres[1] = Sphere(vec3(0.0, 0.0, -70.0), 15.0, { vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), 0.2, 0.8, 0.8, 200.0 }, false);
	scenes[5].spheres[2] = Sphere(vec3(20.0, 0.0, -120.0), 15.0, { vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0), 0.2, 0.8, 0.8, 200.0 }, false);
	scenes[5].spheres[3] = Sphere(vec3(-20.0, 0.0, -120.0), 15.0, { vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), 0.2, 0.8, 0.8, 200.0 }, false);

	scenes[5].planes = new Plane[scenes[5].planeCount];
	cudaMallocManaged((void**)&scenes[5].planes, scenes[5].planeCount * sizeof(Plane));
	scenes[5].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[5].AABBs = new AABB[scenes[5].AABBCount];
	cudaMallocManaged((void**)&scenes[5].AABBs, scenes[5].AABBCount * sizeof(AABB));

	scenes[5].models = new Model[scenes[5].modelCount];
	cudaMallocManaged((void**)&scenes[5].models, scenes[5].modelCount * sizeof(Model));

	scenes[5].lights = new Light[scenes[5].lightCount];
	cudaMallocManaged((void**)&scenes[5].lights, scenes[5].lightCount * sizeof(Light));
	scenes[5].lights[0] = { { -29, -32, 0 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[5].cam, sizeof(Camera));


	// scene 7 - single chest model
	scenes[6].cam = Camera(vec3(48.0561, -33.9654, 32.7296), 90.0, width / (float)height, -146.2, 19.7);
	scenes[6].sphereCount = 1; scenes[6].planeCount = 1; scenes[6].AABBCount = 0; scenes[6].modelCount = 1; scenes[6].lightCount = 1;

	scenes[6].spheres = new Sphere[scenes[6].sphereCount];
	cudaMallocManaged((void**)&scenes[6].spheres, scenes[6].sphereCount * sizeof(Sphere));
	scenes[6].spheres[0] = Sphere(vec3(21, -13, -18), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scenes[6].planes = new Plane[scenes[6].planeCount];
	cudaMallocManaged((void**)&scenes[6].planes, scenes[6].planeCount * sizeof(Plane));
	scenes[6].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[6].AABBs = new AABB[scenes[6].AABBCount];
	cudaMallocManaged((void**)&scenes[6].AABBs, scenes[6].AABBCount * sizeof(AABB));

	scenes[6].models = new Model[scenes[6].modelCount];
	cudaMallocManaged((void**)&scenes[6].models, scenes[6].modelCount * sizeof(Model));
	scenes[6].models[0] = Model(vec3(0.0, 0.0, 0.0), 15, "models/Chest_gold.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.9, 200.0 }, false, Diffuse);

	scenes[6].lights = new Light[scenes[6].lightCount];
	cudaMallocManaged((void**)&scenes[6].lights, scenes[6].lightCount * sizeof(Light));
	scenes[6].lights[0] = { { 21, -13, -18 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[6].cam, sizeof(Camera));

	// scene 8 - reflective boxes + spheres
	scenes[7].cam = Camera(vec3(0.774152, -77.2231, 82.8138), 90.0, width / (float)height, -89.5, 28.7);
	scenes[7].sphereCount = 4; scenes[7].planeCount = 1; scenes[7].AABBCount = 2; scenes[7].modelCount = 0; scenes[7].lightCount = 1;

	scenes[7].spheres = new Sphere[scenes[7].sphereCount];
	cudaMallocManaged((void**)&scenes[7].spheres, scenes[7].sphereCount * sizeof(Sphere));
	scenes[7].spheres[0] = Sphere(vec3(2, -73, -16), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);
	scenes[7].spheres[1] = Sphere(vec3(0.0, 0.0, 0.0), 15.0, { vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), 0.2, 0.8, 0.8, 200.0 }, false, Reflect);
	scenes[7].spheres[2] = Sphere(vec3(0.0, 0.0, -40.0), 15.0, { vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), 0.2, 0.8, 0.8, 200.0 }, false, Reflect);
	scenes[7].spheres[3] = Sphere(vec3(0.0, 0.0, 40.0), 15.0, { vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), 0.2, 0.8, 0.8, 200.0 }, false, Reflect);

	scenes[7].planes = new Plane[scenes[7].planeCount];
	cudaMallocManaged((void**)&scenes[7].planes, scenes[7].planeCount * sizeof(Plane));
	scenes[7].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[7].AABBs = new AABB[scenes[7].AABBCount];
	cudaMallocManaged((void**)&scenes[7].AABBs, scenes[7].AABBCount * sizeof(AABB));
	scenes[7].AABBs[0] = AABB(vec3(-100.0, 0.0, 0.0), 100.0, { vec3(0.2, 0.2, 0.2), vec3(0.2, 0.2, 0.2), 0.5, 0.2, 0.3, 10 }, false, Reflect);
	scenes[7].AABBs[1] = AABB(vec3(100.0, 0.0, 0.0), 100.0, { vec3(0.2, 0.2, 0.2), vec3(0.2, 0.2, 0.2), 0.5, 0.2, 0.3, 10 }, false, Reflect);

	scenes[7].models = new Model[scenes[7].modelCount];
	cudaMallocManaged((void**)&scenes[7].models, scenes[7].modelCount * sizeof(Model));

	scenes[7].lights = new Light[scenes[7].lightCount];
	cudaMallocManaged((void**)&scenes[7].lights, scenes[7].lightCount * sizeof(Light));
	scenes[7].lights[0] = { { 2, -73, -16 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[7].cam, sizeof(Camera));

	// scene 9 - reflective box + model
	scenes[8].cam = Camera(vec3(38.9299, -39.5555, 76.1236), 90.0, width / (float)height, -129.6, 15);
	scenes[8].sphereCount = 1; scenes[8].planeCount = 1; scenes[8].AABBCount = 1; scenes[8].modelCount = 3; scenes[8].lightCount = 1;

	scenes[8].spheres = new Sphere[scenes[8].sphereCount];
	cudaMallocManaged((void**)&scenes[8].spheres, scenes[8].sphereCount * sizeof(Sphere));
	scenes[8].spheres[0] = Sphere(vec3(2, -73, -16), 2.0, { vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0), 1.0, 0.0, 0.0, 200.0 }, true);

	scenes[8].planes = new Plane[scenes[8].planeCount];
	cudaMallocManaged((void**)&scenes[8].planes, scenes[8].planeCount * sizeof(Plane));
	scenes[8].planes[0] = Plane(vec3(0.0, 20.0, 0.0), vec3(0.0, 1.0, 0.0), { vec3(0.1, 0.1, 0.1), vec3(0.1, 0.1, 0.1), 0.5, 0.5, 0.0, 200 }, false, Reflect);

	scenes[8].AABBs = new AABB[scenes[8].AABBCount];
	cudaMallocManaged((void**)&scenes[8].AABBs, scenes[8].AABBCount * sizeof(AABB));
	scenes[8].AABBs[0] = AABB(vec3(-100.0, 0.0, 0.0), 100.0, { vec3(0.2, 0.2, 0.2), vec3(0.2, 0.2, 0.2), 0.5, 0.2, 0.3, 10 }, false, Reflect);

	scenes[8].models = new Model[scenes[8].modelCount];
	cudaMallocManaged((void**)&scenes[8].models, scenes[8].modelCount * sizeof(Model));
	scenes[8].models[0] = Model(vec3(0.0, 19.0, 0.0), 15, "models/Table.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.0, 10.0 }, false, Diffuse);
	scenes[8].models[1] = Model(vec3(0.0, 6.0, 10.0), 10, "models/Chalice.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 1.0, 1000.0 }, false, Reflect);
	scenes[8].models[2] = Model(vec3(0.0, 6.0, 0.0), 10, "models/Book2_Open.obj", { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), 0.1, 0.9, 0.0, 10.0 }, false, Diffuse);

	scenes[8].lights = new Light[scenes[8].lightCount];
	cudaMallocManaged((void**)&scenes[8].lights, scenes[8].lightCount * sizeof(Light));
	scenes[8].lights[0] = { { 2, -73, -16 }, vec3(1.0, 1.0, 1.0), vec3(0.5, 0.0, 0.0), 4, vec3(0.0, 0.0, 0.5), 2, 8 };

	cudaMallocManaged((void**)&scenes[8].cam, sizeof(Camera));


	////////////////////////////////////////////////////////////////////////////////////////////

	cudaMallocManaged((void**)&framebuffer, 3 * width * height * sizeof(GLubyte));

	cudaMallocManaged((void**)&randStates, dimBlock.x * dimBlock.y * dimGrid.x * dimGrid.y * sizeof(curandState));
	setupCurand<<<dimGrid, dimBlock>>>(randStates, unsigned(time(NULL)));
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