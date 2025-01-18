#pragma once

#include "utils.h"

#include "glm\glm.hpp"

class Object {
public:
	virtual bool hit(glm::vec3 rayOrigin, glm::vec3 rayDir, float& t0, float& t1) = 0;
	virtual glm::vec3 getPosition() = 0;
	virtual Material getMaterial() = 0;
	virtual glm::vec3 getNormal(glm::vec3 point) = 0;
	virtual void move() = 0;
};

