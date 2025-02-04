#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "glm\glm.hpp"
#include <gl/glew.h>

#include "vec3.h"

struct Light {
	glm::vec3 position;
	glm::vec3 colour;
	glm::vec3 uvec;
	int usteps;
	glm::vec3 vvec;
	int vsteps;
	int samples;
};

struct Material {
	glm::vec3 colour;
	float ambient;
	float diffuse;
	float specular;
	float shininess;
};

struct RayHit {
	float t = 999;
	Material mat = {};
	glm::vec3 hitPoint = glm::vec3();
	glm::vec3 normal = glm::vec3();
	glm::vec3 objectPos;
};

__host__ __device__ struct CUDALight {
	vec3 position;
	vec3 colour;
	vec3 uvec;
	int usteps;
	vec3 vvec;
	int vsteps;
	int samples;
};

__host__ __device__ struct CUDAMaterial {
	vec3 colour;
	float ambient;
	float diffuse;
	float specular;
	float shininess;
};

__host__ __device__ struct Hit {
	float t = 999;
	CUDAMaterial mat = {};
	vec3 hitPoint;
	vec3 normal;
	vec3 objectPos;
};

__host__ __device__ bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1);
__host__ __device__ vec3 reflect(vec3 a, vec3 b);

template <typename T>
__host__ __device__ T min(T x, T y) {
	return (x < y) ? x : y;
}

template <typename T>
__host__ __device__ T max(T x, T y) {
	return (x > y) ? x : y;
}

__host__ __device__ float inline radians(float angle) {
	return angle * 0.01745329251994329576923690768489;
}