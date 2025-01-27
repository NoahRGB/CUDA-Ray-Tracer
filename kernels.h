#pragma once

#include "vec3.h"
#include "CUDASphere.h"
#include "Camera.h"

#include <cuda_runtime.h>

__global__ void rayTrace(int width, int height, GLubyte* framebuffer, CUDASphere* objects, int objectCount, CUDALight* lights, Camera cam);
