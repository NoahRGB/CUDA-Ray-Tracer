#include "Camera.h"

#include "utils.h"

Camera::Camera(vec3 position, float fov, float aspectRatio) {
	up = vec3(0.0, 1.0, 0.0);
	direction = vec3(0.0, 0.0, 0.0);
	this->position = position;
	this->fov = fov;
	fovFactor = tan(radians(fov / 2));
	this->aspectRatio = aspectRatio;

	pitch = 0;
	yaw = -90;

	moveSpeed = 3.0;
	sensitivity = 0.1;

	updateDirection();
}

__device__ __host__ vec3 Camera::rasterToCameraSpace(float x, float y, int width, int height)  {
	// convert point defined in pixel space to normalised device coordinates
	// (coordinates in the range [0, 1])
	vec3 ndcPoint = vec3(x / width, y / height, 0);

	// convert point in ndc space into screen space, coordinates in range [-1, 1]
	vec3 screenPoint = vec3(2 * ndcPoint.x() - 1, 1 - 2 * ndcPoint.y(), 0);

	// convert point in screen space to camera space using FOV and aspect ratio
	//float fovFactor = tan(radians(fov / 2));
	vec3 cameraPoint = vec3(screenPoint.x() * 1 * fovFactor, screenPoint.y() * fovFactor, 1);

	// use direction vectors to reach final coordinate   
	vec3 final = cameraPoint.x() * right + cameraPoint.y() * up + cameraPoint.z() * direction;

	return final;
}

void Camera::mouseMovement(double xOff, double yOff) {
	yaw += xOff * sensitivity;
	pitch += yOff * sensitivity;

	if (pitch > 89.0f) {
		pitch = 89.0f;
	}
	if (pitch < -89.0f) {
		pitch = -89.0f;
	}

	updateDirection();
}

__device__ __host__ void Camera::updateDirection() {
	float pitchR = radians(pitch);
	float yawR = radians(yaw);

	direction = vec3(cos(yawR) * cos(pitchR), sin(pitchR), sin(yawR) * cos(pitchR));
	direction = normalise(direction);

	right = normalise(cross(direction, vec3(0.0, 1.0, 0.0)));
	up = normalise(cross(right, direction));
}

void Camera::move(CameraMove moveType) {
	if (moveType == FORWARD) {
		position += direction * moveSpeed;
	}
	else if (moveType == BACKWARD) {
		position -= direction * moveSpeed;
	}
	else if (moveType == LEFT) {
		position -= right * moveSpeed;
	}
	else if (moveType == RIGHT) {
		position += right * moveSpeed;
	}
	else if (moveType == UP) {
		position[1] -= moveSpeed;
	}
	else if (moveType == DOWN) {
		position[1] += moveSpeed;
	}
}