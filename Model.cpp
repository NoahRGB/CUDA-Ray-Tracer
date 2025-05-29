#include "Model.h"
#include "OBJ_Loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>


Model::Model() {

}

Model::Model(vec3 position, int size, char* filename, Material mat, bool debug, ObjectType objectType) {
	this->position = position;
	this->size = size;
	this->mat = mat;
	this->debug = debug;
	this->objectType = objectType;
	objectName = Model_t;
	boundingBox = AABB();

	objl::Loader loader;
	loader.LoadFile(filename);

	vertCount = loader.LoadedVertices.size();
	indicesCount = loader.LoadedIndices.size();
	materialCount = loader.LoadedMaterials.size();
	meshCount = loader.LoadedMeshes.size();

	std::cout << std::endl << std::endl;

	std::cout << "Loaded vert count: " << vertCount << std::endl;
	std::cout << "Loaded indices count: " << indicesCount << std::endl;
	std::cout << "Loaded material count: " << materialCount << std::endl;
	std::cout << "Loaded mesh count: " << meshCount << std::endl;

	std::cout << std::endl << std::endl;

	cudaMallocManaged((void**)&vertices, vertCount * sizeof(Vertex));
	cudaMallocManaged((void**)&indices, indicesCount * sizeof(int));

	int x, y, n;
	std::string name = std::string("models/Atlas.png");
	unsigned char* imageData = stbi_load(name.c_str(), &x, &y, &n, 4);

	//set indices
	for (int i = 0; i < indicesCount; i++) {
		indices[i] = loader.LoadedIndices[i];
	}

	// set vertices
	int vertIndex = 0;
	for (int i = 0; i < meshCount; i++) {
		objl::Mesh mesh = loader.LoadedMeshes[i];
		for (int j = 0; j < mesh.Vertices.size(); j++) {

			// copy over the vertex's position/texutre/normal
			vertices[vertIndex].position = vec3(position.x() + mesh.Vertices[j].Position.X * (float)size, position.y() + -mesh.Vertices[j].Position.Y * (float)size, position.z() + mesh.Vertices[j].Position.Z * (float)size);
			vertices[vertIndex].textureCoords = vec3(mesh.Vertices[j].TextureCoordinate.X, mesh.Vertices[j].TextureCoordinate.Y, 0);
			vertices[vertIndex].normal = vec3(mesh.Vertices[j].Normal.X, -mesh.Vertices[j].Normal.Y, mesh.Vertices[j].Normal.Z);

			// extend the bounding box for the model
			boundingBox.extendBy(vertices[vertIndex].position);

			// find the ambient/diffuse/specular colours for the vertex
			vertices[vertIndex].ambient = vec3(mesh.MeshMaterial.Ka.X, mesh.MeshMaterial.Ka.Y, mesh.MeshMaterial.Ka.Z);
			vertices[vertIndex].diffuse = vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z);
			vertices[vertIndex].specular = vec3(mesh.MeshMaterial.Ks.X, mesh.MeshMaterial.Ks.Y, mesh.MeshMaterial.Ks.Z);
			
			if (mesh.MeshMaterial.map_Kd != "") {
				if (imageData != nullptr && x > 0 && y > 0) {
					int desiredX = x * vertices[vertIndex].textureCoords.x();
					int desiredY = y * vertices[vertIndex].textureCoords.y();
					int imageIndex = (desiredY * x) + desiredX;
					vertices[vertIndex].diffuse = vec3(static_cast<int>(imageData[imageIndex]), static_cast<int>(imageData[imageIndex + 1]), static_cast<int>(imageData[imageIndex + 2]));
				}
			}

			std::cout << "Completed vertex: " << vertIndex << " / " << vertCount << std::endl;
			vertIndex++;
		}
	}

	stbi_image_free(imageData);
}

Model::~Model() {

}

__host__ __device__ bool Model::triangleIntersect(vec3 v0, vec3 v1, vec3 v2, vec3 rayOrigin, vec3 rayDir, float& t, float& u, float& v) {
	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;
	vec3 pvec = cross(rayDir, v0v2);
	float det = dot(v0v1, pvec);

	if (fabs(det) <= 0) return false;

	float invDet = 1 / det;

	vec3 tvec = rayOrigin - v0;
	u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	vec3 qvec = cross(tvec, v0v1);
	v = dot(rayDir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;


	t = dot(v0v2, qvec) * invDet;

	return true;
}

__host__ __device__ bool Model::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1, Vertex& hitVertex, RayType rayType, bool accelerate) {

	if (accelerate) {
		if (boundingBox.hit(rayOrigin, rayDir, t0, t1)) {
			float tClosest = 999;
			float u, v, w;
			Vertex v0, v1, v2;

			// for every 3 indices (1 triangle)
			for (int i = 0; i < indicesCount - 3; i += 3) {
				float t_tmp, u_tmp, v_tmp;
				if (triangleIntersect(vertices[indices[i]].position, vertices[indices[i + 1]].position, vertices[indices[i + 2]].position, rayOrigin, rayDir, t_tmp, u_tmp, v_tmp)) {
					if (t_tmp < tClosest && t_tmp > 0) {
						tClosest = t_tmp;
						v0 = vertices[indices[i]];
						v1 = vertices[indices[i + 1]];
						v2 = vertices[indices[i + 2]];
						u = u_tmp;
						v = v_tmp;
					}
				}
			}

			// if an intersection was found
			if (tClosest != 999) {
				float w = 1 - u - v;
				// use u/v/w to interpolate the texture/colours for the hit
				hitVertex = {
					vec3(),
					u * v0.textureCoords + v * v1.textureCoords + w * v2.textureCoords,
					u * v0.normal + v * v1.normal + w * v2.normal,
					v0.materialIndex,
					u * v0.ambient + v * v1.ambient + w * v2.ambient,
					u * v0.diffuse + v * v1.diffuse + w * v2.diffuse,
					u * v0.specular + v * v1.specular + w * v2.specular,
				};
				t0 = tClosest;
				return true;
			}
		}

		return false;
	}


	else {
		float tClosest = 999;
		float u, v, w;
		Vertex v0, v1, v2;

		// for every 3 indices (1 triangle)
		for (int i = 0; i < indicesCount - 3; i += 3) {
			float t_tmp, u_tmp, v_tmp;
			if (triangleIntersect(vertices[indices[i]].position, vertices[indices[i + 1]].position, vertices[indices[i + 2]].position, rayOrigin, rayDir, t_tmp, u_tmp, v_tmp)) {
				if (t_tmp < tClosest) {
					tClosest = t_tmp;
					v0 = vertices[indices[i]];
					v1 = vertices[indices[i + 1]];
					v2 = vertices[indices[i + 2]];
					u = u_tmp;
					v = v_tmp;
				}
			}
		}

		// if an intersection was found
		if (tClosest != 999) {
			float w = 1 - u - v;
			// use u/v/w to interpolate the texture/colours for the hit
			hitVertex = {
				vec3(),
				u * v0.textureCoords + v * v1.textureCoords + w * v2.textureCoords,
				u * v0.normal + v * v1.normal + w * v2.normal,
				v0.materialIndex,
				u * v0.ambient + v * v1.ambient + w * v2.ambient,
				u * v0.diffuse + v * v1.diffuse + w * v2.diffuse,
				u * v0.specular + v * v1.specular + w * v2.specular,
			};
			t0 = tClosest;
			return true;
		}

		return false;
	}
	
}

__host__ __device__ vec3 Model::normalAt(vec3 point) {
	return vec3(1.0, 1.0, 1.0);
}