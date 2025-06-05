#include "OctreeNode.h"

OctreeNode::OctreeNode() {
	isLeaf = true;
}

void OctreeNode::add(Vertex* vertices, int count) {
	verticesCount = count;
	includedVertices = vertices;

	for (int i = 0; i < count; i++) {
		boundingBox.extendBy(vertices[i].position);
	}
}
