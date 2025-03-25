#pragma once

#include "Object.h"


class AABB : public Object {

private:



public:
	AABB();
	AABB(vec3 pos, float size, Material mat, bool debug = false, ObjectType objectType = Diffuse);
	AABB(vec3 min, vec3 max, Material mat, bool debug = false, ObjectType objectType = Diffuse);

	__host__ __device__ void extendBy(vec3 point);
	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);
	__host__ __device__ vec3 normalAt(vec3 point);

	vec3 min, max;
};

