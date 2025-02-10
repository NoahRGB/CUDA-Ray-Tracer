#pragma once

#include "Object.h"

class Plane : public Object {
private:



public:
	Plane();
	Plane(vec3 p, vec3 n, CUDAMaterial mat, bool debug = false);

	vec3 n;

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	__host__ __device__ vec3 normalAt(vec3 point);


};

