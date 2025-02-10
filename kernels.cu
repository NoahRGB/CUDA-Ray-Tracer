#include "kernels.h"
#include "Object.h"
#include "Camera.h"

#include <stdio.h>

#include <device_launch_parameters.h>

__device__ vec3 lighting(CUDAMaterial mat, vec3 lightPos, vec3 lightIntensity, vec3 point, vec3 eye, vec3 normal, SceneConfig& config) {
	vec3 colour;
	vec3 L = normalise(lightPos - point);
	
	// ambient
	if (config.ambientLighting) {
		colour = lightIntensity * mat.colour * mat.ambient;
	}

	// diffuse
	if (config.diffuseLighting) {
		float NdotL = max(dot(normal, L), 0.0f);
		colour += (NdotL > 0.0) ? (mat.colour * lightIntensity * mat.diffuse * NdotL) : vec3();
	}

	// specular
	if (config.specularLighting) {
		vec3 v = normalise(eye);
		vec3 r = normalise(-reflect(L, normal));
		float RdotV = max(dot(r, v), 0.0f);
		colour += mat.specular * lightIntensity * pow(RdotV, mat.shininess);
	}

	return colour;
}

template<typename T> __device__ bool traceRay(T* objects, int objectCount, vec3 origin, vec3 dir, RayType rayType, Hit& hit, vec3 ignore = vec3(999, 999, 999)) {
	bool shadowHit = false;
	for (int i = 0; i < objectCount; i++) {
			float t0, t1;
			if (objects[i].hit(origin, dir, t0, t1)) {

				if (rayType == ShadowRay) {
					if (!objects[i].debug) return true;
					continue;
				}

				//if (rayType == ShadowRay) {
				//	if (objects[i].position == ignore) return false;
				//	if (!objects[i].debug) shadowHit = true;
				//	continue;
				//}

				//else {
				if (t0 < hit.t) {
					vec3 hitPoint = origin + t0 * dir;
					vec3 normal = objects[i].normalAt(hitPoint);
					hit = { t0, objects[i].mat, hitPoint, normal, objects[i].position };
				}
				//}
			}
	}

	if (shadowHit) {
		return true;
	}

	if (hit.t != 999) {
		hit.normal = normalise(hit.normal);
		return true;
	}

	return false;
}

__device__ vec3 rayCast(vec3& origin, vec3& dir, Scene& scene, SceneConfig& config, int depth = 1) {
	Hit sphereHit, planeHit, closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, PrimaryRay, sphereHit);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, PrimaryRay, planeHit);
	closestHit = (sphereHit.t < planeHit.t) ? sphereHit : planeHit;

	if (sphereTrace || planeTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);
	
		if (config.reflections) {

		}

		if (config.renderHardShadows) {
			Hit shadowHit;
			bool shadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit, closestHit.objectPos);
			return col * !shadowTrace;
		}

		return col;
	}

	return vec3(0.4, 0.4, 0.4);
}


__global__ void rayTrace(int width, int height, GLubyte* framebuffer, Scene scene, SceneConfig config) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelIndex = y * width + x;
	if (x >= width || y >= height) return;

	vec3 cameraSpacePoint = scene.cam.rasterToCameraSpace(float(x + 0.5), float(y + 0.5), width, height);

	vec3 col = rayCast(scene.cam.getPosition(), normalise(cameraSpacePoint), scene, config);

	framebuffer[pixelIndex * 3 + 0] = min(col.x(), 1.0f) * 255;
	framebuffer[pixelIndex * 3 + 1] = min(col.y(), 1.0f) * 255;
	framebuffer[pixelIndex * 3 + 2] = min(col.z(), 1.0f) * 255;
}