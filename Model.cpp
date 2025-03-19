#include "Model.h"
#include "OBJ_Loader.h"


Model::Model() {

}

Model::Model(vec3 position, int size, char* filename, Material mat, bool debug, ObjectType objectType) {
	this->position = position;
	this->size = size;
	this->mat = mat;
	this->debug = debug;
	this->objectType = objectType;
	objectName = Model_t;
	bb = BoundingBox();

	objl::Loader loader;
	loader.LoadFile(filename);
	vertCount = loader.LoadedMeshes[0].Vertices.size();

	vertices = new vec3[vertCount];
	cudaMallocManaged((void**)&vertices, vertCount * sizeof(vec3));

	for (int i = 0; i < vertCount; i++) {
		objl::Mesh mesh = loader.LoadedMeshes[0];
		vertices[i] = vec3(mesh.Vertices[i].Position.X * (float)size, mesh.Vertices[i].Position.Y * (float)size, mesh.Vertices[i].Position.Z * (float)size);
		bb.extendBy(position + vertices[i]);
	}
}

__host__ __device__ bool Model::triangleIntersect(vec3 v0, vec3 v1, vec3 v2, vec3 rayOrigin, vec3 rayDir, float& t) {
	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;
	vec3 pvec = cross(rayDir, v0v2);
	float det = dot(v0v1, pvec);

	if (fabs(det) <= 0) return false;

	float invDet = 1 / det;

	vec3 tvec = rayOrigin - v0;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	vec3 qvec = cross(tvec, v0v1);
	float v = dot(rayDir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = dot(v0v2, qvec) * invDet;

	return true;
}

__host__ __device__ bool Model::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1) {

	if (bb.hit(rayOrigin, rayDir, t0, t1)) {
		for (int i = 0; i < vertCount - 3; i += 3) {
			if (triangleIntersect(position + vertices[i], position + vertices[i + 1], position + vertices[i + 2], rayOrigin, rayDir, t0)) {
				return true;
			}
		}
	}

	return false;
}

__host__ __device__ vec3 Model::normalAt(vec3 point) {
	return vec3(1.0, 1.0, 1.0);
}