#pragma once

#include "vec3.h"
#include "utils.h"
#include "Object.h"

class CUDASphere : public Object {
private:


public:
	CUDASphere();
	CUDASphere(vec3 center, float radius, CUDAMaterial mat, bool debug = false);

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	float radius;

	__host__ __device__ vec3 normalAt(vec3 point);

};

