#pragma once

#include "vec3.h"
#include "utils.h"

class Camera {

private:

	vec3 up;
	vec3 right;
	vec3 direction;
	vec3 position;

	float fov;
	float near, far;

	float moveSpeed;
	float sensitivity;


public:
	enum CameraMove {
		FORWARD, BACKWARD,
		LEFT, RIGHT,
		UP, DOWN
	};

	float pitch;
	float yaw;

	Camera(vec3 position, float fov);

	__device__ __host__ vec3 rasterToCameraSpace(float x, float y, int width, int height);
	__device__ __host__ void updateDirection();
	void mouseMovement(double xOff, double yOff);
	void move(CameraMove moveType);

	__device__ __host__ vec3 getPosition() { return position; }
	__device__ __host__ float getMoveSpeed() { return moveSpeed; }
	__device__ __host__ float getSensitivity() { return sensitivity; }

};

