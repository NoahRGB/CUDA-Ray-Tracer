#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "glm\glm.hpp"
#include <gl/glew.h>

#include "vec3.h"

__host__ __device__ enum ObjectType {
	Diffuse,
	Reflect,
};

__host__ __device__ enum RayType {
	PrimaryRay,
	ShadowRay,
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
	ObjectType objectType = Diffuse;
};

__host__ __device__ bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1);
__host__ __device__ vec3 lightingReflect(vec3& a, vec3& b);

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