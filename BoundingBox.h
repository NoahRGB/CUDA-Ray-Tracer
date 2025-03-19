#pragma once

#include "vec3.h"

class BoundingBox {
private:


public:
	BoundingBox();

	__host__ __device__ void extendBy(vec3 point);
	__host__ __device__ void addBuffer();
	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	vec3 min;
	vec3 max;
};

