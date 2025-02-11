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

template<typename T> __device__ bool traceRay(T* objects, int objectCount, vec3 origin, vec3 dir, RayType rayType, Hit& hit, vec3& ignore = vec3(-999, -999, -999)) {
	for (int i = 0; i < objectCount; i++) {

		if (objects[i].position != ignore) {
			float t0, t1;
			if (objects[i].hit(origin, dir, t0, t1)) {

				if (rayType == ShadowRay) {
					if (!objects[i].debug) return true;
					continue;
				}

				if (t0 < hit.t) {
					vec3 hitPoint = origin + t0 * dir;
					vec3 normal = objects[i].normalAt(hitPoint);
					hit = { t0, objects[i].mat, hitPoint, normal, objects[i].position, objects[i].objectType, objects[i].debug };
				}
			}
		}
	}

	if (hit.t != 999) {
		hit.normal = normalise(hit.normal);
		return true;
	}

	return false;
}

__device__ vec3 reflectionCast2(vec3& origin, vec3& dir, Scene& scene, SceneConfig& config, vec3& ignore, int depth = 1) {
	Hit sphereHit, planeHit, closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, PrimaryRay, sphereHit, ignore);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, PrimaryRay, planeHit, ignore);
	closestHit = (sphereHit.t < planeHit.t) ? sphereHit : planeHit;

	if (sphereTrace || planeTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);

		if (config.renderHardShadows) {
			Hit shadowHit;
			bool shadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
			return col * !shadowTrace;
		}

		return col;
	}

	return config.backgroundCol;
}


__device__ vec3 reflectionCast(vec3& origin, vec3& dir, Scene& scene, SceneConfig& config, vec3& ignore, int depth = 1) {
	Hit sphereHit, planeHit, closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, PrimaryRay, sphereHit, ignore);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, PrimaryRay, planeHit, ignore);
	closestHit = (sphereHit.t < planeHit.t) ? sphereHit : planeHit;

	if (sphereTrace || planeTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);

		if (config.reflections && closestHit.objectType == Reflect && depth <= config.maxDepth) {
			vec3 r = normalise(lightingReflect(dir, closestHit.normal));
			vec3 reflectionCol = reflectionCast2(closestHit.hitPoint + closestHit.normal * config.shadowBias, r, scene, config, closestHit.objectPos, depth++);
			col += config.reflectionStrength * reflectionCol;
		}

		if (config.renderHardShadows) {
			Hit shadowHit;
			bool shadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
			return col * !shadowTrace;
		}

		return col;
	}

	return config.backgroundCol;
}

__device__ vec3 rayCast(vec3& origin, vec3& dir, Scene& scene, SceneConfig& config) {
	Hit sphereHit, planeHit, closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, PrimaryRay, sphereHit);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, PrimaryRay, planeHit);
	closestHit = (sphereHit.t < planeHit.t) ? sphereHit : planeHit;

	if (sphereTrace || planeTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);

		if (!closestHit.debug) {
			if (config.reflections && closestHit.objectType == Reflect) {
				vec3 r = normalise(lightingReflect(dir, closestHit.normal));
				vec3 reflectionCol = reflectionCast(closestHit.hitPoint + closestHit.normal * config.shadowBias, r, scene, config, closestHit.objectPos);
				col += config.reflectionStrength * reflectionCol;
			}

			if (config.renderHardShadows) {
				Hit shadowHit;
				bool shadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
				col = col * !shadowTrace;
			}
		}

		return col;
	}

	return config.backgroundCol;
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