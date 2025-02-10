#pragma once

#include "utils.h"
#include "vec3.h"

#include <cuda_runtime.h>

class Object {

private:


public:

	vec3 position;
	CUDAMaterial mat;
	bool debug;

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);
	__host__ __device__ vec3 normalAt(vec3 point);


};