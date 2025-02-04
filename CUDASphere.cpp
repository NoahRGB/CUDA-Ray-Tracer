#include "CUDASphere.h"
#include "utils.h"

CUDASphere::CUDASphere() {
	position = vec3(0.0, 0.0, 0.0);
}

CUDASphere::CUDASphere(vec3 center, float radius, CUDAMaterial mat) {
	this->position = center;
	this->radius = radius;
	this->mat = mat;
}

__host__ __device__ bool CUDASphere::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1) {
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

	if (tmpT0 > tmpT1) {
		float temp = tmpT1;
		tmpT1 = tmpT0;
		tmpT0 = temp;
	}

	t0 = tmpT0; t1 = tmpT1;
	return true;
}

__host__ __device__ vec3 CUDASphere::normalAt(vec3 point) {
	return point - position;
}