#pragma once

#include "CUDASphere.h"
#include "Camera.h"
#include "vec3.h"

#include <gl/glew.h>
#include <map>

class RayTracer {
public:
	CUDASphere* objects;
	int objectCount;

	GLubyte* framebuffer;

	CUDALight* lights;

	Camera cam;

	int width, height;

	RayTracer();
	~RayTracer();

	void init(int width, int height);

	void initialiseScene();
	void launchKernel();
	void resize(int width, int height);
};

