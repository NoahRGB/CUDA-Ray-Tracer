#pragma once

#include "Object.h"


class Triangle : public Object {
private:

	vec3 a, b, c;

public:
	Triangle();
	Triangle(vec3 a, vec3 b, vec3 c, Material mat, bool debug = false, ObjectType objectType = Diffuse);
	
	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	__host__ __device__ vec3 normalAt(vec3 point);

};

