#pragma once

#include "Object.h"



class Box : public Object {
private:

	vec3 min, max;


public:
	Box();
	Box(vec3 pos, float size, Material mat, bool debug = false, ObjectType objectType = Diffuse);
	Box(vec3 min, vec3 max, Material mat, bool debug = false, ObjectType objectType = Diffuse);

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	__host__ __device__ vec3 normalAt(vec3 point);

};

