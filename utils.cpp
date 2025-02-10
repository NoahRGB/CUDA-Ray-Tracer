#include "utils.h"

#include <cmath>
#include <numbers>
#include <algorithm>

#include "glm\glm.hpp"

__host__ __device__ bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1) {
	float discriminant = b * b - 4 * a * c;

	if (discriminant < 0) {
		return false;
	}
	else if (discriminant == 0) {
		x0 = x1 = -0.5 * b / a;
	}
	else {
		float q = (b > 0) ? -0.5 * (b + sqrt(discriminant)) : -0.5 * (b - sqrt(discriminant));
		x0 = q / a;
		x1 = c / q;
	}

	//if (x0 > x1) std::swap(x0, x1);
	if (x0 > x1) {
		float temp = x0;
		x0 = x1;
		x1 = temp;
	}
	return true;
}

__host__ __device__ vec3 lightingReflect(vec3 a, vec3 b) {
	return a - 2 * dot(a, b) * b;
}