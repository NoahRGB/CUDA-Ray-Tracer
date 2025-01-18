#define GLM_FORCE_CUDA
#include "Sphere.h"

Sphere::Sphere(glm::vec3 center, float radius, glm::vec3 matCol, float matAmbient, float matDiffuse, float matSpecular, float matShininess) {
	this->center = center;
	this->radius = radius;
	mat = { matCol, matAmbient, matDiffuse, matSpecular, matShininess };
}

Sphere::Sphere(glm::vec3 center, float radius, Material mat) {
	this->center = center;
	this->radius = radius;
	this->mat = mat;
}

bool Sphere::hit(glm::vec3 rayOrigin, glm::vec3 rayDir, float& t0, float& t1) {
	float tmpT0, tmpT1;

	glm::vec3 L = rayOrigin - center;
	float a = glm::dot(rayDir, rayDir);
	float b = 2 * glm::dot(rayDir, L);
	float c = glm::dot(L, L) - radius * radius;
	if (!solveQuadratic(a, b, c, tmpT0, tmpT1)) return false;


	if (tmpT0 > tmpT0) std::swap(tmpT0, tmpT0);
	t0 = tmpT0; t1 = tmpT1;
	return true;
}

glm::vec3 Sphere::getNormal(glm::vec3 point) {
	return point - center;
}

void Sphere::move() {
	center.y += 10.0;
}