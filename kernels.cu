#include "kernels.h"
#include "Camera.h"

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

	// specular
	//vec3 v = unit_vector(eye);
	//vec3 r = unit_vector(-reflect(L, normal));
	//float RdotV = dot(r, v);
	//if (RdotV < 0) RdotV = 0;

	//vec3 specular = mat.specular * lightIntensity * pow(RdotV, mat.shininess);
	//colour += specular;

	return colour;
}

__global__ void rayTrace(int width, int height, GLubyte* framebuffer, CUDASphere* objects, CUDALight* lights, Camera cam) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelIndex = y * width + x;
	if (x >= width || y >= height) return;


	vec3 cameraSpacePoint = cam.rasterToCameraSpace(float(x + 0.5), float(y + 0.5), 1000, 1000);
	float t0, t1;
	if (objects[0].hit(cam.getPosition(), normalise(cameraSpacePoint), t0, t1)) {

		float smallest = t0;
		if (t1 < smallest) smallest = t1;

		vec3 hitPoint = cam.getPosition() + smallest * normalise(cameraSpacePoint);


		vec3 col = lighting(objects[0].mat, lights[0].position, lights[0].colour, hitPoint, cam.getPosition() + vec3(1.0, 1.0, 1.0), normalise(objects[0].normalAt(hitPoint)));

		if (col.x() > 1) col = vec3(1, col.y(), col.z());
		if (col.y() > 1) col = vec3(col.x(), 1, col.z());
		if (col.z() > 1) col = vec3(col.x(), col.y(), 1);

		framebuffer[pixelIndex * 3 + 0] = 255 * col.x();
		framebuffer[pixelIndex * 3 + 1] = 255 * col.y();
		framebuffer[pixelIndex * 3 + 2] = 255 * col.z();

	} else {
		// must be a background pixel
		framebuffer[pixelIndex * 3 + 0] = 0.0;
		framebuffer[pixelIndex * 3 + 1] = 0.0;
		framebuffer[pixelIndex * 3 + 2] = 0.0;
	}
}