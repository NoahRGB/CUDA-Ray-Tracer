#include "RayTracer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <iostream>

RayTracer::RayTracer(int width, int height, glm::vec3 backgroundCol, float fov) {
	this->width = width;
	this->height = height;

	this->fov = fov;
	bg = backgroundCol;
	aspectRatio = width / (float)height;
}

RayTracer::~RayTracer() {
	std::cout << "Destructing objects vector..." << std::endl;
	for (Object* obj : objects) {
		delete obj;
	}
}

RayHit RayTracer::rayCast(glm::vec3 rayOrigin, glm::vec3 rayDir) {
	RayHit closestHit;

	for (Object* object : objects) {
		float t0, t1;
		if (object->hit(rayOrigin, rayDir, t0, t1)) {
			if (t0 >= 0 && t1 >= 0) {
				float smallest = std::min(t0, t1);
				if (smallest < closestHit.t) {
					glm::vec3 hitPoint = rayOrigin + smallest * rayDir;
					glm::vec3 normal = object->getNormal(hitPoint);
					normal = glm::normalize(normal);

					closestHit = { smallest, object->getMaterial(), hitPoint, normal, object->getPosition() };
				}
			}
		}
	}
	return closestHit;
}

bool RayTracer::inShadow(glm::vec3 point) {
	//for (Light light : lights) {
	//	glm::vec3 dir = glm::normalize(light.position - point);
	//	RayHit hit = rayCast(point + dir, dir);
	//	if (hit.hitPoint != glm::vec3(0, 0, 0)) {
	//		return true;
	//	}
	//}
	return false;
}

void RayTracer::hardShadow(glm::vec3 point, glm::vec3* colourOut) {
	if (inShadow(point)) {
		colourOut->x = 0;
		colourOut->y = 0;
		colourOut->z = 0;
	}
}

void RayTracer::softShadow(glm::vec3 point, glm::vec3* colourOut) {
	//for (Light light : lights) {
	//	float intensity = 0.0;
	//	for (int v = 0; v < light.vsteps; v++) {
	//		for (int u = 0; u < light.usteps; u++) {
	//			glm::vec3 lightPos = light.position + light.uvec * (float)(u + 0.5) + light.vvec * (float)(v + 0.5);
	//			glm::vec3 dir = glm::normalize(lightPos - point);

	//			if (rayCast(point, dir).hitPoint == glm::vec3(0, 0, 0)) {
	//				intensity += 1.0;
	//			}
	//		}
	//		intensity = intensity / light.samples;

	//		colourOut->x *= (intensity * 2);
	//		colourOut->y *= (intensity * 2);
	//		colourOut->z *= (intensity * 2);
	//	}
	//}
}

__device__ void RayTracer::rayTrace(vec3* framebuffer, bool softShadows) {

	for (int x = 0; x < 1000; x++) {
		for (int y = 0; y < 1000; y++) {
			int pixelIndex = y * width + x;
			//pixels[x][y][0] = bg.x;
			//pixels[x][y][1] = bg.y;
			//pixels[x][y][2] = bg.z;
			framebuffer[pixelIndex] = vec3(bg.x, bg.y, bg.z);

			//glm::vec3 totalPixelColour = glm::vec3(0.0, 0.0, 0.0);
			//int pixelsSampled = 0;
			//bool backgroundPixel = true;

			//// for every sample in the pixel
			//for (float pixelOffset = 0.5; pixelOffset < 1.0; pixelOffset += 0.5) {

			//	// calculate camera space and detect the closest hit object
			//	glm::vec3 cameraSpacePoint = rasterToCameraSpace(float(x + pixelOffset), float(y + pixelOffset));
			//	RayHit closestHit = rayCast(glm::vec3(0.0, 0.0, 0.0), glm::normalize(cameraSpacePoint));

			//	if (closestHit.hitPoint != glm::vec3(0, 0, 0)) {

			//		backgroundPixel = false;

			//		// calculate colour at this sample from lighting and the closest hit
			//		glm::vec3 sampleColour = glm::vec3(0.0, 0.0, 0.0);
			//		for (Light light : lights) {
			//			glm::vec3 lightCol = lighting(closestHit.mat, light.position, light.colour, closestHit.hitPoint, glm::vec3(1, 1, 1), closestHit.normal);
			//			lightCol *= 255;
			//			sampleColour += lightCol;
			//		}
			//		
			//		// add hard or soft shadow effects
			//		//if (softShadows) {
			//		//	softShadow(closestHit.hitPoint, &sampleColour);
			//		//}
			//		//else {
			//		//	hardShadow(closestHit.hitPoint, &sampleColour);
			//		//}

			//		totalPixelColour += sampleColour;
			//		pixelsSampled++;
			//	}
			//}

			//if (!backgroundPixel) {

			//	// average pixel colour over the number of sampled pixels
			//	totalPixelColour /= pixelsSampled;

			//	// clamp max colour value for each channel at 255
			//	if (totalPixelColour.x > 255) totalPixelColour.x = 255;
			//	if (totalPixelColour.y > 255) totalPixelColour.y = 255;
			//	if (totalPixelColour.z > 255) totalPixelColour.z = 255;

			//	// set pixel colour
			//	pixels[x][y][0] = totalPixelColour.x;
			//	pixels[x][y][1] = totalPixelColour.y;
			//	pixels[x][y][2] = totalPixelColour.z;
			//}
		}
	}
}

void RayTracer::addSphere(glm::vec3 pos, float radius, Material mat) {
	objects.push_back(new Sphere(pos, radius, mat));
}

void RayTracer::addLight(glm::vec3 pos, glm::vec3 colour, glm::vec3 uvec, int usteps, glm::vec3 vvec, int vsteps, int samples) {
	lights.push_back({ pos, colour, uvec, usteps, vvec, vsteps, samples });
}

glm::vec3 RayTracer::lighting(Material mat, glm::vec3 lightPos, glm::vec3 lightIntensity, glm::vec3 point, glm::vec3 eye, glm::vec3 normal) {
	glm::vec3 L = glm::normalize(lightPos - point);
	float NdotL = glm::dot(normal, L);
	if (NdotL < 0) NdotL = 0;

	glm::vec3 colour = lightIntensity * mat.colour * mat.ambient;

	glm::vec3 v = glm::normalize(eye);
	glm::vec3 r = glm::normalize(-glm::reflect(L, normal));
	float RdotV = glm::dot(r, v);
	if (RdotV < 0) RdotV = 0;

	if (NdotL > 0.0) {
		glm::vec3 diffuse = mat.colour * lightIntensity * mat.diffuse * NdotL;
		colour += diffuse;
	}


	glm::vec3 specular = mat.specular * lightIntensity * glm::pow(RdotV, mat.shininess);

	colour += specular;
	return colour;
}

glm::vec3 RayTracer::rasterToCameraSpace(float x, float y) {
	// convert point defined in pixel space to normalised device coordinates
	// (coordinates in the range [0, 1])
	glm::vec2 ndcPoint = glm::vec2(x / width, y / height);

	// convert point in ndc space into screen space (coordinates in range [-1, 1]
	glm::vec2 screenPoint = glm::vec2(2 * ndcPoint.x - 1, 1 - 2 * ndcPoint.y);

	// convert point in screen space to camera space using FOV and aspect ratio
	float fovFactor = glm::radians(fov / 2);
	glm::vec2 cameraPoint = glm::vec2(screenPoint.x * aspectRatio * fovFactor, screenPoint.y * fovFactor);

	// final camera space coordinate is (camPointX, camPointY, -1)
	return glm::vec3(cameraPoint, -1);
}

__device__ void RayTracer::rayTraceTest(vec3** framebuffer) {
	for (int x = 0; x < 1000; x++) {
		for (int y = 0; y < 1000; y++) {
	//		int pixelIndex = x * (width-1) + y;

	//		*framebuffer[pixelIndex] = vec3(100, 100, 100);
		}
	}
}

//glm::mat4x4 cameraToWorld;
//glm::vec3 rayOriginWorld = cameraToWorld * rayOrigin;
//glm::vec4 test = glm::vec4(cameraSpacePoint, 1.0);
//std::cout << test.x << " " << test.y << " " << test.z << std::endl;
//glm::vec3 rayDirWorld = glm::vec4(cameraSpacePoint, 1.0) * cameraToWorld;
//glm::vec3 rayDirWorld = glm::normalize(rayDirWorld);
//rayDirWorld = rayDirWorld - rayOriginWorld;