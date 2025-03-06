#pragma once

#include "vec3.h"
#include "utils.h"
#include "Object.h"

class Sphere : public Object {
private:


public:
	Sphere();
	Sphere(vec3 center, float radius, Material mat, bool debug = false, ObjectType objectType = Diffuse);

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	float radius;

	__host__ __device__ vec3 normalAt(vec3 point);

};

