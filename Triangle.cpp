#include "Triangle.h"



Triangle::Triangle() {

}

Triangle::Triangle(vec3 a, vec3 b, vec3 c, Material mat, bool debug, ObjectType objectType) {
	this->a = a;
	this->b = b;
	this->c = c;
	this->mat = mat;
	this->debug = debug;
	this->objectType = objectType;
	objectName = Triangle_t;
}


__host__ __device__ bool Triangle::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1) {
	vec3 v0v1 = b - a;
	vec3 v0v2 = c - a;
	vec3 pvec = cross(rayDir, v0v2);
	float det = dot(v0v1, pvec);

	if (fabs(det) < 0.001) return false;

	float invDet = 1 / det;

	vec3 tvec = rayOrigin - a;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	vec3 qvec = cross(tvec, v0v1);
	float v = dot(rayDir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t0 = dot(v0v2, qvec) * invDet;

	return true;

}


__host__ __device__ vec3 Triangle::normalAt(vec3 point) {
	return vec3(1.0, 1.0, 1.0);
}