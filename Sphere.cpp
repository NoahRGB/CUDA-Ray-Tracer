#include "Sphere.h"
#include "utils.h"

Sphere::Sphere() {
	position = vec3(0.0, 0.0, 0.0);
}

Sphere::Sphere(vec3 center, float radius, Material mat, bool debug, ObjectType objectType) {
	this->position = center;
	this->radius = radius;
	this->mat = mat;
	this->debug = debug;
	this->objectType = objectType;
	objectName = Sphere_t;
}

__host__ __device__ bool Sphere::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1) {
	float tmpT0, tmpT1;

	vec3 L = rayOrigin - position;
	float a = dot(rayDir, rayDir);
	float b = 2 * dot(rayDir, L);
	float c = dot(L, L) - radius * radius;
	if (!solveQuadratic(a, b, c, tmpT0, tmpT1)) return false;

	if (tmpT0 < 0) {
		tmpT0 = tmpT1;
		if (tmpT0 < 0) return false;
	}

	if (tmpT1 < 0) {
		tmpT1 = tmpT0;
		if (tmpT1 < 0) return false;
	}

	if (tmpT0 > tmpT1) {
		float temp = tmpT1;
		tmpT1 = tmpT0;
		tmpT0 = temp;
	}

	t0 = tmpT0; t1 = tmpT1;
	return true;
}

__host__ __device__ vec3 Sphere::normalAt(vec3 point) {
	return point - position;
}