#pragma once

#include "Object.h"
#include "vec3.h"
#include "AABB.h"
#include "Octree.h"

class BV;

class Model : public Object {

private:
	__host__ __device__ bool triangleIntersect(vec3 v0, vec3 v1, vec3 v2, vec3 rayOrigin, vec3 rayDir, float& t, float& u, float& v, bool cull);

public:
	Model();
	~Model();
	Model(vec3 position, int size, char* filename, Material mat, bool debug = false, ObjectType objectType = Diffuse);

	__host__ __device__ bool hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1, Vertex& hitVertex, RayType rayType, bool accelerate=false, bool accelerateTwice=false, bool useOctree=true, bool cull=true);

	__host__ __device__ vec3 normalAt(vec3 point);

	__host__ __device__ void setupBoundingBoxes(int verticesPerBB);
	__host__ __device__ void extendBoundingBoxes();

	vec3 position;
	int size;

	AABB boundingBox;

	int bbCount = 8;
	AABB boundingBoxes[8];
	Octree octree;

	Vertex* vertices;
	int vertCount;

	int materialCount;
	int meshCount;

	int* indices;
	int indicesCount;
};

