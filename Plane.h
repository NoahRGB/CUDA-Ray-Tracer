#pragma once

#include "Object.h"
#include "utils.h"

class Plane : public Object {
private:



public:
	Plane();
	Plane(vec3 p, vec3 n, Material mat, bool debug = false, ObjectType objectType = Diffuse);

	vec3 n;

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	__host__ __device__ vec3 normalAt(vec3 point);


};

