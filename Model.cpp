#include "Model.h"
#include "OBJ_Loader.h"


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

	cudaMallocManaged((void**)&materials, materialCount * sizeof(TextureMaterial));
	cudaMallocManaged((void**)&vertices, vertCount * sizeof(Vertex));
	cudaMallocManaged((void**)&indices, indicesCount * sizeof(int));


	// set materials
	for (int i = 0; i < materialCount; i++) {
		objl::Material mat = loader.LoadedMaterials[i];
		materials[i].name = (char*)mat.name.c_str();
		materials[i].ambient = vec3(mat.Ka.X, mat.Ka.Y, mat.Ka.Z);
		materials[i].diffuse = vec3(mat.Kd.X, mat.Kd.Y, mat.Kd.Z);
		materials[i].specular = vec3(mat.Ks.X, mat.Ks.Y, mat.Ks.Z);
	}

	//set indices
	for (int i = 0; i < indicesCount; i++) {
		indices[i] = loader.LoadedIndices[i];
	}

	// set vertices
	int vertIndex = 0;
	for (int i = 0; i < meshCount; i++) {
		objl::Mesh mesh = loader.LoadedMeshes[i];
		for (int j = 0; j < mesh.Vertices.size(); j++) {

			vertices[vertIndex].position = vec3(position.x() + mesh.Vertices[j].Position.X * (float)size, position.y() + mesh.Vertices[j].Position.Y * (float)size, position.z() + mesh.Vertices[j].Position.Z * (float)size);
			vertices[vertIndex].textureCoords = vec3(mesh.Vertices[j].TextureCoordinate.X, mesh.Vertices[j].TextureCoordinate.Y, 0);
			vertices[vertIndex].normal = vec3(mesh.Vertices[j].Normal.X, mesh.Vertices[j].Normal.Y, mesh.Vertices[j].Normal.Z);

			boundingBox.extendBy(vertices[vertIndex].position);

			// find corresponding material
			for (int k = 0; k < materialCount; k++) {
				if (materials[k].diffuse == vec3(mesh.MeshMaterial.Kd.X, mesh.MeshMaterial.Kd.Y, mesh.MeshMaterial.Kd.Z)) {
					vertices[vertIndex].materialIndex = k;
					break;
				}
			}
			vertIndex++;
		}
	}
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

__host__ __device__ bool Model::hit(vec3 rayOrigin, vec3 rayDir, float& t0, float& t1, Vertex& hitVertex) {

	if (boundingBox.hit(rayOrigin, rayDir, t0, t1)) {
		//for (int i = 0; i < vertCount - 3; i += 3) {
		//	float u, v;
		//	if (triangleIntersect(vertices[i].position, vertices[i + 1].position, vertices[i + 2].position, rayOrigin, rayDir, t0, u, v)) {
		//		hitVertex = {
		//			vec3(),
		//			vertices[i].textureCoords,
		//			vertices[i].normal,
		//			vertices[i].materialIndex,
		//		};
		//		return true;
		//	}
		//}

		float tClosest = 999;
		float u, v, w;
		Vertex v0, v1, v2;

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

		if (tClosest != 999) {
			float w = 1 - u - v;
			hitVertex = {
				vec3(),
				u * v0.textureCoords + v * v1.textureCoords + v2.textureCoords,
				u * v0.normal + v * v1.normal + w * v2.normal,
				v0.materialIndex,
			};
			t0 = tClosest;
			return true;
		}
	}

	return false;
}

__host__ __device__ vec3 Model::normalAt(vec3 point) {
	return vec3(1.0, 1.0, 1.0);
}