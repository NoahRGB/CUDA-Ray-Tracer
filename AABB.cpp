#include "AABB.h"

AABB::AABB() {
	min = vec3(0.0, 0.0, 0.0);
	max = vec3(0.0, 0.0, 0.0);
}

AABB::AABB(vec3 pos, float size, Material mat, bool debug, ObjectType objectType) {
	this->min = pos - (size / 2);
	this->max = pos + (size / 2);
	this->mat = mat;
	this->debug = debug;
	this->objectType = objectType;
	objectName = AABB_t;
}

AABB::AABB(vec3 min, vec3 max, Material mat, bool debug, ObjectType objectType) {
	this->min = min;
	this->max = max;
	this->mat = mat;
	this->debug = debug;
	this->objectType = objectType;
	objectName = AABB_t;
}

__host__ __device__ void AABB::extendBy(vec3 point) {
	if (point.x() < min.x()) min.nums[0] = point.x() - 0.1;
	if (point.y() < min.y()) min.nums[1] = point.y() - 0.1;
	if (point.z() < min.z()) min.nums[2] = point.z() - 0.1;

	if (point.x() > max.x()) max.nums[0] = point.x() + 0.1;
	if (point.y() > max.y()) max.nums[1] = point.y() + 0.1;
	if (point.z() > max.z()) max.nums[2] = point.z() + 0.1;
}

__host__ __device__ bool AABB::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1) {
	// get tmin_x and tmax_x and put them in the correct order
	float tmin = (min.x() - rayOrigin.x()) / rayDir.x();
	float tmax = (max.x() - rayOrigin.x()) / rayDir.x();
	if (tmin > tmax) swap(tmin, tmax);

	// get tmin_y and tmax_y and put them in the correct order
	float tmin_y = (min.y() - rayOrigin.y()) / rayDir.y();
	float tmax_y = (max.y() - rayOrigin.y()) / rayDir.y();
	if (tmin_y > tmax_y) swap(tmin_y, tmax_y);

	// ray can't intersect with box if tmin > tmax_y or tmin_y > tmax
	if ((tmin > tmax_y) || (tmin_y > tmax)) return false;

	// update tmin/tmax so it refers to the closest intersection
	if (tmin_y > tmin) tmin = tmin_y;
	if (tmax_y < tmax) tmax = tmax_y;

	// get tmin_z and tmax_z and put them in the correct order
	float tmin_z = (min.z() - rayOrigin.z()) / rayDir.z();
	float tmax_z = (max.z() - rayOrigin.z()) / rayDir.z();
	if (tmin_z > tmax_z) swap(tmin_z, tmax_z);

	// ray can't intersect with box if tmin > tmax_z or tmin_z > tmax
	if ((tmin > tmax_z) || (tmin_z > tmax)) return false;

	// update tmin/tmax so it refers to the closest intersection
	if (tmin_z > tmin) tmin = tmin_z;
	if (tmax_z < tmax) tmax = tmax_z;

	// don't allow negatives
	if (tmin < 0) {
		tmin = tmax;
		if (tmin < 0) return false;
	}

	if (tmax < 0) {
		tmax = tmin;
		if (tmax < 0) return false;
	}

	// make sure the order is correct
	if (tmin > tmax) swap(tmin, tmax);

	t0 = tmin;
	t1 = tmax;

	return true;
}

__host__ __device__ vec3 AABB::normalAt(vec3 point) {
	vec3 boxCenter = (min + max) / 2;
	vec3 p = point - boxCenter;

	vec3 d = abs(min - max) / 2;
	float bias = 1.00001;
	vec3 normal = normalise(vec3(int(p.x() / d.x() * bias), int(p.y() / d.y() * bias), int(p.z() / d.z() * bias)));
	//printf("%f, %f, %f\n", normal.x(), normal.y(), normal.z());
	return normal;
}

__host__ __device__ void AABB::setupIndices(int indicesCount) {
	this->indicesCount = indicesCount;
	cudaMallocManaged((void**)&includedModelIndices, indicesCount * sizeof(int));
	cudaDeviceSynchronize();
}