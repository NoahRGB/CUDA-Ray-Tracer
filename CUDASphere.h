#pragma once

#include "vec3.h"
#include "utils.h"

class CUDASphere {
private:




public:
	CUDASphere();
	CUDASphere(vec3 center, float radius, CUDAMaterial mat);

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	vec3 center;
	float radius;
	CUDAMaterial mat;

	__host__ __device__ vec3 normalAt(vec3 point);

};

