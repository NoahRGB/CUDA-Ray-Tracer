#pragma once

#include "vec3.h"
#include "CUDASphere.h"
#include "Plane.h"
#include "Camera.h"
#include "RayTracer.h"

#include <cuda_runtime.h>

__global__ void rayTrace(int width, int height, GLubyte* framebuffer, Scene scene, SceneConfig config);
