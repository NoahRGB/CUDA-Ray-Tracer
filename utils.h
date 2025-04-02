#pragma once

#include <vector>
#include <random>
#include <string>

#include <cuda_runtime.h>

#include "glm\glm.hpp"
#include <gl/glew.h>

#include "vec3.h"

__host__ __device__ enum ObjectName {
	Sphere_t, Plane_t, Box_t, Triangle_t, Model_t, AABB_t
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
	vec3 ambientColour;
	vec3 diffuseColour;
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

__host__ __device__ struct Vertex {
	vec3 position;
	vec3 textureCoords;
	vec3 normal;
	int materialIndex;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

__host__ __device__ struct TextureMaterial {
	std::string name = "";
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	std::string ambient_map_name = "";
	std::string diffuse_map_name = "";
	std::string specular_map_name = "";
};

__host__ __device__ bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1);
__host__ __device__ vec3 abs(vec3& v);
float getRand(int min, int max);
float getNormalRand(std::mt19937& gen, std::normal_distribution<float>& normal);

vec3* generatePlaneSetNormals();

template <typename T> __host__ __device__ T min(T x, T y) {
	return (x < y) ? x : y;
}


template <typename T> __host__ __device__ T max(T x, T y) {
	return (x > y) ? x : y;
}


template <typename T> __host__ __device__ void swap(T& a, T& b) {
	T temp = a;
	a = b;
	b = temp;
}

__host__ __device__ float inline radians(float angle) {
	return angle * 0.01745329251994329576923690768489;
}