#define GLM_FORCE_CUDA
#pragma once

#include "Object.h"
#include "utils.h"

#include "glm\glm.hpp"

class Sphere : public Object {
private:
	Material mat;
	glm::vec3 center;
	float radius;

public:
	Sphere() {}
	Sphere(glm::vec3 center, float radius, glm::vec3 matCol, float matAmbient, float matDiffuse, float matSpecular, float matShininess);
	Sphere(glm::vec3 center, float radius, Material mat);
	bool hit(glm::vec3 rayOrigin, glm::vec3 rayDir, float& t0, float& t1);

	glm::vec3 getPosition() { return center; }
	Material getMaterial() { return mat; }
	glm::vec3 getNormal(glm::vec3 point);

	void move();
};

