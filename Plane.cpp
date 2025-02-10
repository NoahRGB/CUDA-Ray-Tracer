#include "Plane.h"

Plane::Plane() {

}
Plane::Plane(vec3 p, vec3 n, CUDAMaterial mat, bool debug, ObjectType objectType) {
	position = p;
	this->mat = mat;
	this->n = n;
	this->debug = false;
	this->objectType = objectType;
}

__host__ __device__ bool Plane::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1) {
	float denominator = dot(n, rayDir);
	vec3 v = position - rayOrigin;
	t0 = dot(v, n) / denominator;
	return (t0 >= 0);
	return false;
}

__host__ __device__ vec3 Plane::normalAt(vec3 point) {
	return n;
}