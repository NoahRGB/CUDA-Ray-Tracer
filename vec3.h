#pragma once

#include <math.h>
#include <cuda_runtime.h>

class vec3 {

public:
	__host__ __device__ vec3() { nums[0] = 0; nums[1] = 0; nums[2] = 0; }
	__host__ __device__ vec3(float x, float y, float z) { nums[0] = x; nums[1] = y; nums[2] = z; }

	__host__ __device__ inline float x() const { return nums[0]; }
	__host__ __device__ inline float r() const { return nums[0]; }
	__host__ __device__ inline float y() const { return nums[1]; }
	__host__ __device__ inline float g() const { return nums[1]; }
	__host__ __device__ inline float z() const { return nums[2]; }
	__host__ __device__ inline float b() const { return nums[2]; }

	__host__ __device__ inline vec3 operator-() { return vec3(-nums[0], -nums[1], -nums[2]); }
	__host__ __device__ inline float operator[](int i) const { return nums[i]; }
	__host__ __device__ inline float& operator[](int i) { return nums[i]; }

	__host__ __device__ inline float length() const { return sqrt(nums[0] * nums[0] + nums[1] * nums[1] + nums[2] * nums[2]); }

	__host__ __device__ inline vec3& operator+=(const vec3& v2);
	__host__ __device__ inline vec3& operator+=(int num);
	__host__ __device__ inline vec3& operator-=(const vec3& v2);
	__host__ __device__ inline bool operator!=(const vec3& v2);
	__host__ __device__ inline bool operator==(const vec3& v2);

private:
	float nums[3];

};

__host__ __device__ inline bool vec3::operator==(const vec3& v2) {
	return (nums[0] == v2.x()) && (nums[1] == v2.y()) && (nums[2] == v2.z());
}

__host__ __device__ inline bool vec3::operator!=(const vec3& v2) {
	return !((nums[0] == v2.x()) && (nums[1] == v2.y()) && (nums[2] == v2.z()));
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v2) {
	nums[0] = nums[0] + v2.x();
	nums[1] = nums[1] + v2.y();
	nums[2] = nums[2] + v2.z();
	return *this;
}

__host__ __device__ inline vec3& vec3::operator+=(int num) {
	nums[0] = nums[0] + num;
	nums[1] = nums[1] + num;
	nums[2] = nums[2] + num;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v2) {
	nums[0] = nums[0] - v2.x();
	nums[1] = nums[1] - v2.y();
	nums[2] = nums[2] - v2.z();
	return *this;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
	return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

__host__ __device__ inline vec3 operator+(const vec3& v1, float n) {
	return vec3(v1[0] + n, v1[1] + n, v1[2] + n);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
	return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
	return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float n) {
	return vec3(v[0] * n, v[1] * n, v[2] * n);
}

__host__ __device__ inline vec3 operator*(float n, const vec3& v) {
	return vec3(v[0] * n, v[1] * n, v[2] * n);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
	return vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v, float n) {
	return vec3(v[0] / n, v[1] / n, v[2] / n);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y() * v2.z() - v1.z() * v2.y(),
		v1.z() * v2.x() - v1.x() * v2.z(),
		v1.x() * v2.y() - v1.y() * v2.x());
}

__host__ __device__ inline vec3 normalise(vec3 v) {
	return v / v.length();
}

__host__ __device__ inline vec3 reflect(vec3 a, vec3 b) {
	return a - b * dot(a, b) * vec3(2.0, 2.0, 2.0);
}