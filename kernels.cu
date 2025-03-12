#include "kernels.h"
#include "Object.h"
#include "Camera.h"

#include <stdio.h>
#include <math.h>

#include <device_launch_parameters.h>
#include <curand_kernel.h>

__device__ float wangHash(unsigned int seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return float(seed) / 4294967296.0f;
}

__device__ vec3 lighting(Material mat, vec3 lightPos, vec3 lightIntensity, vec3 point, vec3 eye, vec3 normal, SceneConfig& config) {
	vec3 colour;
	vec3 L = normalise(lightPos - point);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
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
		if (config.areaLightSpecularEffect) {
			for (int i = 0; i < config.softShadowNum; i++) {

				unsigned int seed = (x * 1000 + y) * 1973 + i * 9277;
				float radius = config.softShadowRadius;
				float theta = 2 * 3.141592654f * wangHash(seed);
				float phi = acos(2 * wangHash(seed * 16807) - 1);
				vec3 pointOnSphere = vec3(radius * sin(phi) * cos(theta), radius * sin(phi) * sin(theta), radius * cos(phi));
				vec3 L = normalise((lightPos + pointOnSphere) - point);

				vec3 v = normalise(eye);
				vec3 r = normalise(-reflect(L, normal));
				float RdotV = max(dot(r, v), 0.0f);
				colour += mat.specular * lightIntensity * pow(RdotV, mat.shininess);

			}
		}
		else {
			vec3 v = normalise(eye);
			vec3 r = normalise(-reflect(L, normal));
			float RdotV = max(dot(r, v), 0.0f);
			colour += mat.specular * lightIntensity * pow(RdotV, mat.shininess);
		}


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

				if (rayType == ReflectRay && objects[i].debug) continue;

				if (t0 < hit.t) {
					vec3 hitPoint = origin + t0 * dir;
					vec3 normal = objects[i].normalAt(hitPoint);
					hit = { t0, objects[i].mat, hitPoint, normal, objects[i].position, objects[i].objectType, objects[i].objectName, objects[i].debug };
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
	Hit sphereHit, planeHit, boxHit, closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, ReflectRay, sphereHit);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, ReflectRay, planeHit);
	bool boxTrace = traceRay(scene.boxes, scene.boxCount, origin, dir, ReflectRay, boxHit);
	closestHit = (sphereHit.t < planeHit.t) ? ((sphereHit.t < boxHit.t) ? sphereHit : boxHit) : ((planeHit.t < boxHit.t) ? planeHit : boxHit);


	if (sphereTrace || planeTrace || boxTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);

		if (config.renderHardShadows) {
			Hit shadowHit;
			bool shadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
			col = shadowTrace ? col * config.shadowIntensity : col;
		}

		return col;
	}

	return config.backgroundCol;
}

__device__ vec3 reflectionCast(vec3& origin, vec3& dir, Scene& scene, SceneConfig& config, vec3& ignore, int depth = 1) {
	Hit sphereHit, planeHit, boxHit, closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, ReflectRay, sphereHit);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, ReflectRay, planeHit);
	bool boxTrace = traceRay(scene.boxes, scene.boxCount, origin, dir, ReflectRay, boxHit);
	closestHit = (sphereHit.t < planeHit.t) ? ((sphereHit.t < boxHit.t) ? sphereHit : boxHit) : ((planeHit.t < boxHit.t) ? planeHit : boxHit);


	if (sphereTrace || planeTrace || boxTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);

		if (config.reflections && closestHit.objectType == Reflect && depth <= config.maxDepth) {
			vec3 r = normalise(reflect(dir, closestHit.normal));
			vec3 reflectionCol = reflectionCast2(closestHit.hitPoint + closestHit.normal * config.shadowBias, r, scene, config, closestHit.objectPos, depth++);
			if (closestHit.objectName == ObjectName::Plane_t) {
				col += config.planeReflectionStrength * reflectionCol;
			}
			else if (closestHit.objectName == ObjectName::Sphere_t) {
				col += config.sphereReflectionStrength * reflectionCol;
			}
		}

		if (config.renderHardShadows) {
			Hit shadowHit;
			bool shadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
			col = shadowTrace ? col * config.shadowIntensity : col;
		}

		return col;
	}

	return config.backgroundCol;
}

__device__ vec3 rayCast(vec3& origin, vec3& dir, Scene& scene, SceneConfig& config, curandState randState) {
	Hit closestHit;
	bool sphereTrace = traceRay(scene.spheres, scene.sphereCount, origin, dir, PrimaryRay, closestHit);
	bool planeTrace = traceRay(scene.planes, scene.planeCount, origin, dir, PrimaryRay, closestHit);
	bool boxTrace = traceRay(scene.boxes, scene.boxCount, origin, dir, PrimaryRay, closestHit);
	bool triangleTrace = traceRay(scene.triangles, scene.triangleCount, origin, dir, PrimaryRay, closestHit);
	bool modelTrace = traceRay(scene.models, scene.modelCount, origin, dir, PrimaryRay, closestHit);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (sphereTrace || planeTrace || boxTrace || triangleTrace || modelTrace) {
		vec3 col = lighting(closestHit.mat, scene.lights[0].position, scene.lights[0].colour, closestHit.hitPoint, scene.cam.getPosition(), normalise(closestHit.normal), config);

		if (!closestHit.debug) {
			if (config.reflections && closestHit.objectType == Reflect) {
				vec3 r = normalise(reflect(dir, closestHit.normal));
				vec3 reflectionCol = reflectionCast(closestHit.hitPoint + closestHit.normal * config.shadowBias, r, scene, config, closestHit.objectPos);
				
				if (closestHit.objectName == ObjectName::Plane_t) {
					col += config.planeReflectionStrength * reflectionCol;
				}
				else if (closestHit.objectName == ObjectName::Sphere_t) {
					col += config.sphereReflectionStrength * reflectionCol;
				}
				else if (closestHit.objectName == ObjectName::Box_t) {
					col += config.boxReflectionStrength * reflectionCol;
				}
			}

			if (config.renderHardShadows) {
				Hit shadowHit;
				bool sphereShadowTrace = traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
				bool boxShadowTrace = traceRay(scene.boxes, scene.boxCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(scene.lights[0].position - closestHit.hitPoint), ShadowRay, shadowHit);
				col = (sphereShadowTrace || boxShadowTrace) ? col * config.shadowIntensity : col;
			}

			if (config.renderSoftShadows) {
				int hits = 0;
				for (int i = 0; i < config.softShadowNum; i++) {
					
					// generate random points on a unit sphere
					unsigned int seed = (x * 1000 + y) * 1973 + i * 9277;
					float r = config.softShadowRadius;
					float theta = 2 * 3.141592654f * wangHash(seed);
					float phi = acos(2 * wangHash(seed * 16807) - 1);
					vec3 pointOnSphere = vec3(r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi));

					// add random points to the light position and test for collision
					vec3 lightPoint = scene.lights[0].position + pointOnSphere;
					Hit hit;
					hits += traceRay(scene.spheres, scene.sphereCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(lightPoint - closestHit.hitPoint), ShadowRay, hit) ? 1 : 0;
					hits += traceRay(scene.boxes, scene.boxCount, closestHit.hitPoint + closestHit.normal * config.shadowBias, normalise(lightPoint - closestHit.hitPoint), ShadowRay, hit) ? 1 : 0;

				}
				col = col * (1 - ((float)hits / config.softShadowNum));
			}
		}

		return col;
	}

	return vec3(0.1, 0.1, 0.1) * (float)config.backgroundBrightness;
}

__global__ void rayTrace(int width, int height, GLubyte* framebuffer, Scene scene, SceneConfig config, curandState* randStates) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelIndex = y * width + x;
	if (x >= width || y >= height) return;

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state = randStates[id];

	vec3 col;

	if (config.antiAliasing) {
		vec3 total;
		vec3 cameraSpacePoint;
		for (float i = 0.25; i <= 0.75; i += 0.5) {
			for (float j = 0.25; j <= 0.75; j += 0.5) {
				cameraSpacePoint = scene.cam.rasterToCameraSpace(float(x + i), float(y + j), width, height);
				total += rayCast(scene.cam.getPosition(), normalise(cameraSpacePoint), scene, config, state);
			}
		}
		col = total / 4;
	}
	else {
		vec3 cameraSpacePoint = scene.cam.rasterToCameraSpace(float(x + 0.5), float(y + 0.5), width, height);
		col = rayCast(scene.cam.getPosition(), normalise(cameraSpacePoint), scene, config, state);
	}

	framebuffer[pixelIndex * 3 + 0] = min(col.x(), 1.0f) * 255;
	framebuffer[pixelIndex * 3 + 1] = min(col.y(), 1.0f) * 255;
	framebuffer[pixelIndex * 3 + 2] = min(col.z(), 1.0f) * 255;
}

__global__ void setupCurand(curandState* randStates) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1337, id, 0, &randStates[id]);
}






//vec3 origin = scene.cam.getPosition();
//vec3 dir = normalise(scene.cam.rasterToCameraSpace(float(x + 0.5), float(y + 0.5), width, height));
//
//vec3 v0 = vec3(0.0, 0.0, 0.0);
//vec3 v1 = vec3(50.0, 0.0, 0.0);
//vec3 v2 = vec3(50.0, 0.0, -50.0);
//
//vec3 AB = v1 - v0;
//vec3 AC = v2 - v0;
//vec3 N = cross(AB, AC);
//
//float rayNormalAngle = dot(N, dir);
//if (abs(rayNormalAngle) < 0.001) {
//	col = config.backgroundCol;
//}
//else {
//	float d = -dot(N, v0);
//	float t = -(dot(N, origin) + d) / rayNormalAngle;
//	if (t < 0) {
//		col = config.backgroundCol;
//	}
//	else {
//		vec3 p = origin + dir * t;
//		vec3 Ne;
//		vec3 v0p = p - v0;
//		Ne = cross(AB, v0p);
//		if (dot(N, Ne) < 0) {
//			col = config.backgroundCol;
//		}
//		else {
//			vec3 CB = v2 - v1;
//			vec3 v1p = p - v1;
//			Ne = cross(CB, v1p);
//			if (dot(N, Ne) < 0) {
//				col = config.backgroundCol;
//			}
//			else {
//				vec3 CA = v0 - v2;
//				vec3 v2p = p - v2;
//				Ne = cross(CA, v2p);
//				if (dot(N, Ne) < 0) {
//					col = config.backgroundCol;
//				}
//				else {
//					col = vec3(1.0, 0.0, 0.0);
//				}
//			}
//		}
//
//	}
//}