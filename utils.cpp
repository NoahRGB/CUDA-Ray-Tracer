#include "utils.h"

#include <iostream>
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

float getRand(int min, int max) {
	return rand() % (max - min + 1) + min;
}

float getNormalRand(std::mt19937& gen, std::normal_distribution<float>& normal) {
	float val = -1;

	while (val < 0.0f || val > 1.0f) {
		val = normal(gen);
	}

	return val;
}

__host__ __device__ vec3 abs(vec3& v) {
	return vec3(abs(v.x()), abs(v.y()), abs(v.z()));
}

vec3* generatePlaneSetNormals() {
	vec3* planeSetNormals = new vec3[7];
	planeSetNormals[0] = vec3(1.0, 0.0, 0.0);
	planeSetNormals[1] = vec3(0.0, 1.0, 0.0);
	planeSetNormals[2] = vec3(0.0, 0.0, 1.0);
	planeSetNormals[3] = vec3(sqrtf(3) / 3.0, sqrtf(3) / 3.0, sqrtf(3) / 3.0);
	planeSetNormals[4] = vec3(-sqrtf(3) / 3.0, sqrtf(3) / 3.0, sqrtf(3) / 3.0);
	planeSetNormals[5] = vec3(-sqrtf(3) / 3.0, -sqrtf(3) / 3.0, sqrtf(3) / 3.0);
	planeSetNormals[6] = vec3(sqrtf(3) / 3.0, -sqrtf(3) / 3.0, sqrtf(3) / 3.0);
	return planeSetNormals;
}