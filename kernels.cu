#include "kernels.h"
#include "Camera.h"

#include <stdio.h>

#include <device_launch_parameters.h>

__device__ vec3 lighting(CUDAMaterial mat, vec3 lightPos, vec3 lightIntensity, vec3 point, vec3 eye, vec3 normal) {

	// ambient
	vec3 colour = lightIntensity * mat.colour * mat.ambient;

	// diffuse
	vec3 L = normalise(lightPos - point);
	float NdotL = dot(normal, L);
	if (NdotL < 0) NdotL = 0;

	if (NdotL > 0.0) {
		vec3 diffuse = mat.colour * lightIntensity * mat.diffuse * NdotL;
		colour += diffuse;
	}

	//colour = (NdotL > 0.0) ? mat.colour * lightIntensity * mat.diffuse * NdotL : 0;

	// specular
	vec3 v = normalise(eye);
	vec3 r = normalise(-reflect(L, normal));
	float RdotV = dot(r, v);
	if (RdotV < 0) RdotV = 0;

	vec3 specular = mat.specular * lightIntensity * pow(RdotV, mat.shininess);
	colour += specular;

	return colour;
}

__device__ Hit rayCast(CUDASphere* objects, int objectCount, vec3 origin, vec3 dir) {
	Hit closestHit;

	for (int i = 0; i < objectCount; i++) {
		//skip object
		float t0, t1;
		CUDASphere ob = objects[i];
		if (objects[i].hit(origin, dir, t0, t1)) {
			if (!(t0 < 0 && t1 < 0)) {
				float smallest;
				if (t0 < 0) {
					smallest = t1;
				} else if (t1 < 0) {
					smallest = t0;
				}
				else {
					smallest = min(t0, t1);
				}

				if (smallest < closestHit.t) {
					vec3 hitPoint = origin + smallest * dir;
					vec3 normal = objects[i].normalAt(hitPoint);
					normal = normalise(normal);//save doing for end

					closestHit = { smallest, objects[i].mat, hitPoint, normal, objects[i].center };
				}
			}
		}
	}

	return closestHit;
}

__device__ bool hardShadow(Hit hit, CUDALight* lights, CUDASphere* objects, int objectCount) {
	for (int i = 0; i < 1; i++) { // for every light
		vec3 dir = normalise(lights[i].position - hit.hitPoint);
		Hit shadowHit = rayCast(objects, objectCount, hit.hitPoint + dir, dir);
		if (shadowHit.hitPoint != vec3(0, 0, 0) && shadowHit.objectPos != hit.objectPos) {
			//Hit test = rayCast(objects, objectCount, hit.hitPoint + dir, dir);
			return true;

		}

		//vec3 dir = normalise(hit.hitPoint - lights[i].position);
		//Hit shadowHit = rayCast(objects, objectCount, lights[i].position + dir, dir);
		//if (shadowHit.hitPoint != vec3(0, 0, 0) && shadowHit.objectPos != hit.objectPos) {

		//	return true;
		//}
	}
	return false;
}

__global__ void rayTrace(int width, int height, GLubyte* framebuffer, CUDASphere* objects, int objectCount, CUDALight* lights, Camera cam) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelIndex = y * width + x;
	if (x >= width || y >= height) return;

	if (pixelIndex > 250000) {
		int a = 10;
	}


	vec3 cameraSpacePoint = cam.rasterToCameraSpace(float(x + 0.5), float(y + 0.5), width, height);
	Hit closestHit = rayCast(objects, objectCount, cam.getPosition(), normalise(cameraSpacePoint));

	if (closestHit.hitPoint != vec3(0, 0, 0)) {

		vec3 col = lighting(closestHit.mat, lights[0].position, lights[0].colour, closestHit.hitPoint, cam.getPosition(), normalise(closestHit.normal));

		if (hardShadow(closestHit, lights, objects, objectCount)) {
			col = vec3(0.0, 0.0, 0.0);
		}

		//min(col.x, 1.0, col.x);

		if (col.x() > 1) col = vec3(1, col.y(), col.z());
		if (col.y() > 1) col = vec3(col.x(), 1, col.z());
		if (col.z() > 1) col = vec3(col.x(), col.y(), 1);

		framebuffer[pixelIndex * 3 + 0] = 255 * col.x();
		framebuffer[pixelIndex * 3 + 1] = 255 * col.y();
		framebuffer[pixelIndex * 3 + 2] = 255 * col.z();
	}
	else {
		// must be a background pixel
		framebuffer[pixelIndex * 3 + 0] = 100.0;
		framebuffer[pixelIndex * 3 + 1] = 100.0;
		framebuffer[pixelIndex * 3 + 2] = 100.0;
	}
}