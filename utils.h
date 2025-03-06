#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "glm\glm.hpp"
#include <gl/glew.h>

#include "vec3.h"

__host__ __device__ enum ObjectName {
	Sphere_t, Plane_t, Box_t
};

__host__ __device__ enum ObjectType {
	Diffuse,
	Reflect,
};

__host__ __device__ enum RayType {
	PrimaryRay,
	ShadowRay,
	ReflectRay
};

__host__ __device__ struct Light {
	vec3 position;
	vec3 colour;
	vec3 uvec;
	int usteps;
	vec3 vvec;
	int vsteps;
	int samples;
};

__host__ __device__ struct Material {
	vec3 colour;
	float ambient;
	float diffuse;
	float specular;
	float shininess;
};

__host__ __device__ struct Hit {
	float t = 999;
	Material mat = {};
	vec3 hitPoint;
	vec3 normal;
	vec3 objectPos;
	ObjectType objectType = Diffuse;
	ObjectName objectName;
	bool debug = false;
};

__host__ __device__ bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1);

float getRand(int min, int max);
bool loadOBJ(const char* path, std::vector<glm::vec3>& vertices, std::vector<glm::vec2>& uvs, std::vector<glm::vec3>& normals);

__host__ __device__ vec3 abs(vec3& v);

template <typename T>
__host__ __device__ T min(T x, T y) {
	return (x < y) ? x : y;
}

template <typename T>
__host__ __device__ T max(T x, T y) {
	return (x > y) ? x : y;
}

template <typename T>
__host__ __device__ void swap(T& a, T& b) {
	T temp = a;
	a = b;
	b = temp;
}

__host__ __device__ float inline radians(float angle) {
	return angle * 0.01745329251994329576923690768489;
}