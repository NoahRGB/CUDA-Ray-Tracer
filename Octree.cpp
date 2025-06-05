#include "Octree.h"

Octree::Octree() {
	root = new OctreeNode();
}

void Octree::init(Vertex* allVertices, int verticesCount) {
	root->add(allVertices, verticesCount);
}

void Octree::build(OctreeNode* currentNode) {
	
	//OctreeNode* currentNode = root;
	
	if (currentNode->verticesCount > 300) {
		currentNode->isLeaf = false;

		cudaMallocManaged((void**)&currentNode->children, 8 * sizeof(OctreeNode));
		cudaDeviceSynchronize();

		int verticesPerChild = (int)std::ceil((float)currentNode->verticesCount / 3.0);
		//int verticesPerChild = ((int)std::ceil(std::ceil((float)currentNode->verticesCount / 3.0) / 8.0)) * 3;
		int currentVertex = 0;

		for (int i = 0; i < 8; i++) {

			Vertex* childrenVertices;
			cudaMallocManaged((void**)&childrenVertices, verticesPerChild * sizeof(Vertex));
			cudaDeviceSynchronize();

			for (int j = 0; j < verticesPerChild; j++) {
				if (currentVertex < currentNode->verticesCount) {
					childrenVertices[j] = currentNode->includedVertices[currentVertex];
					currentVertex++;
				}
			}

			currentNode->children[i].add(childrenVertices, verticesPerChild);

			build(&currentNode->children[i]);
		}

	}
}