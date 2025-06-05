#include "Model.h"
#include "OBJ_Loader.h"
#include "queue.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>


Model::Model() {

}

__host__ __device__ void Model::setupBoundingBoxes(int verticesPerBB) {
	for (int i = 0; i < bbCount; i++) {
		boundingBoxes[i].setupIndices(verticesPerBB);
	}
}

__host__ __device__ void Model::extendBoundingBoxes() {
	for (int i = 0; i < bbCount; i++) {
		for (int j = 0; j < boundingBoxes[i].indicesCount; j++) {
			boundingBoxes[i].extendBy(vertices[boundingBoxes[i].includedModelIndices[j]].position);
		}
	}
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

	int trianglesPerBox = (int)std::ceil(std::ceil(indicesCount / 3) / bbCount);
	int indicesPerBox = trianglesPerBox * 3;
	int currentBoundingBoxIndex = 0;
	int currentBoundingBoxIndicesIndex = 0;
	std::cout << "Indices per box: " << indicesPerBox << std::endl;
	setupBoundingBoxes(indicesPerBox);

	//set indices
	for (int i = 0; i < indicesCount - 3; i += 3) {
		indices[i] = loader.LoadedIndices[i];
		indices[i+1] = loader.LoadedIndices[i+1];
		indices[i+2] = loader.LoadedIndices[i+2];

		boundingBoxes[currentBoundingBoxIndex].includedModelIndices[currentBoundingBoxIndicesIndex] = loader.LoadedIndices[i];
		boundingBoxes[currentBoundingBoxIndex].includedModelIndices[currentBoundingBoxIndicesIndex+1] = loader.LoadedIndices[i+1];
		boundingBoxes[currentBoundingBoxIndex].includedModelIndices[currentBoundingBoxIndicesIndex+2] = loader.LoadedIndices[i+2];

		currentBoundingBoxIndicesIndex += 3;
		if (currentBoundingBoxIndicesIndex >= indicesPerBox) {
			currentBoundingBoxIndex++;
			currentBoundingBoxIndicesIndex = 0;
		}
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

	//Vertex* allVertices = new Vertex[indicesCount];
	//for (int i = 0; i < indicesCount; i++) {
	//	allVertices[i] = vertices[indices[i]];
	//}
	//octree.init(allVertices, indicesCount);
	//octree.build(octree.root);

	extendBoundingBoxes();

	stbi_image_free(imageData);
}

Model::~Model() {

}

__host__ __device__ bool Model::triangleIntersect(vec3 v0, vec3 v1, vec3 v2, vec3 rayOrigin, vec3 rayDir, float& t, float& u, float& v, bool cull) {
	
	vec3 v0v1 = v1 - v0;
	vec3 v0v2 = v2 - v0;
	vec3 pvec = cross(rayDir, v0v2);
	float det = dot(v0v1, pvec);

	if (cull && det >= -0.01) return false;
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

__host__ __device__ bool Model::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1, Vertex& hitVertex, RayType rayType, bool accelerate, bool accelerateTwice, bool useOctree, bool cull) {

	if (useOctree) {

		//Queue queue;
		//queue.enqueue(*octree.root);
		
		//while (queue.used != 0) {
		//	OctreeNode node = queue.dequeue();
		//	float temp_t0, temp_t1;
		//	if (node.boundingBox.hit(rayOrigin, rayDir, temp_t0, temp_t1)) {
		//		if (node.isLeaf) {

		//			float tClosest = 999;
		//			float u, v, w;
		//			Vertex v0, v1, v2;

		//			for (int i = 0; i < node.verticesCount - 3; i += 3) {
		//				float t_tmp, u_tmp, v_tmp;
		//				if (triangleIntersect(node.includedVertices[i].position, node.includedVertices[i+1].position, node.includedVertices[i+2].position, rayOrigin, rayDir, t_tmp, u_tmp, v_tmp, cull)) {
		//					if (t_tmp < tClosest && t_tmp > 0) {
		//						tClosest = t_tmp;
		//						v0 = node.includedVertices[i];
		//						v1 = node.includedVertices[i+1];
		//						v2 = node.includedVertices[i+2];
		//						u = u_tmp;
		//						v = v_tmp;
		//					}
		//				}
		//			}

		//			if (tClosest != 999) {
		//				float w = 1 - u - v;
		//				// use u/v/w to interpolate the texture/colours for the hit
		//				hitVertex = {
		//					vec3(),
		//					u * v0.textureCoords + v * v1.textureCoords + w * v2.textureCoords,
		//					u * v0.normal + v * v1.normal + w * v2.normal,
		//					v0.materialIndex,
		//					u * v0.ambient + v * v1.ambient + w * v2.ambient,
		//					u * v0.diffuse + v * v1.diffuse + w * v2.diffuse,
		//					u * v0.specular + v * v1.specular + w * v2.specular,
		//				};
		//				t0 = tClosest;
		//				return true;
		//			}



		//		} else {
		//			for (int i = 0; i < 8; i++) {
		//				queue.enqueue(node.children[i]);
		//			}
		//		}
		//	}
		//}

		return false;
	}
	else if (accelerateTwice) {

		Vertex currentBestVertex;
		float currentBestT = 999;

		for (int i = 0; i < bbCount; i++) {
			float bb_t0, bb_t1;
			if (boundingBoxes[i].hit(rayOrigin, rayDir, bb_t0, bb_t1)) {

				float tClosest = 999;
				float u, v, w;
				Vertex v0, v1, v2;

				// for every 3 indices (1 triangle)
				for (int j = 0; j < boundingBoxes[i].indicesCount - 3; j += 3) {
					float t_tmp, u_tmp, v_tmp;
					if (triangleIntersect(vertices[boundingBoxes[i].includedModelIndices[j]].position, vertices[boundingBoxes[i].includedModelIndices[j + 1]].position, vertices[boundingBoxes[i].includedModelIndices[j + 2]].position, rayOrigin, rayDir, t_tmp, u_tmp, v_tmp, cull)) {
						if (t_tmp < tClosest && t_tmp > 0) {
							tClosest = t_tmp;
							v0 = vertices[boundingBoxes[i].includedModelIndices[j]];
							v1 = vertices[boundingBoxes[i].includedModelIndices[j + 1]];
							v2 = vertices[boundingBoxes[i].includedModelIndices[j + 2]];
							u = u_tmp;
							v = v_tmp;
						}
					}
				}

				// if an intersection was found
				if (tClosest != 999) {
					float w = 1 - u - v;
					// use u/v/w to interpolate the texture/colours for the hit
					if (tClosest < currentBestT) {
						currentBestT = tClosest;
						currentBestVertex = {
							vec3(),
							u * v0.textureCoords + v * v1.textureCoords + w * v2.textureCoords,
							u * v0.normal + v * v1.normal + w * v2.normal,
							v0.materialIndex,
							u * v0.ambient + v * v1.ambient + w * v2.ambient,
							u * v0.diffuse + v * v1.diffuse + w * v2.diffuse,
							u * v0.specular + v * v1.specular + w * v2.specular,
						};
					}
				}
			}
		}

		if (currentBestT != 999) {
			hitVertex = currentBestVertex;
			t0 = currentBestT;
			return true;
		}
		return false;
	}
	else if (accelerate) {
		if (boundingBox.hit(rayOrigin, rayDir, t0, t1)) {
			float tClosest = 999;
			float u, v, w;
			Vertex v0, v1, v2;

			// for every 3 indices (1 triangle)
			for (int i = 0; i < indicesCount - 3; i += 3) {

	
				float t_tmp, u_tmp, v_tmp;
				if (triangleIntersect(vertices[indices[i]].position, vertices[indices[i + 1]].position, vertices[indices[i + 2]].position, rayOrigin, rayDir, t_tmp, u_tmp, v_tmp, cull)) {
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
			if (triangleIntersect(vertices[indices[i]].position, vertices[indices[i + 1]].position, vertices[indices[i + 2]].position, rayOrigin, rayDir, t_tmp, u_tmp, v_tmp, cull)) {
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