#pragma once

#include "vec3.h"
#include "Sphere.h"
#include "Plane.h"
#include "Camera.h"
#include "RayTracer.h"

#include <cuda_runtime.h>

__global__ void rayTrace(int width, int height, GLubyte* framebuffer, Scene scene, SceneConfig config, curandState* randStates);
__global__ void setupCurand(curandState* randStates, unsigned long seed);