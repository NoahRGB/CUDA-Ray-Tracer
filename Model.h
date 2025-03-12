#pragma once


#include "Object.h"
#include "vec3.h"



class Model : public Object {

private:
	__host__ __device__ bool triangleIntersect(vec3 v0, vec3 v1, vec3 v2, vec3 rayOrigin, vec3 rayDir, float& t);


public:
	Model();
	Model(vec3 position, char* filename, Material mat, bool debug = false, ObjectType objectType = Diffuse);

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1);

	__host__ __device__ vec3 normalAt(vec3 point);

	vec3 position;
	vec3* vertices;
	int vertCount;
	

};

