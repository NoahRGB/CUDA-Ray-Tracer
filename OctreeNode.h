#pragma once

#include "AABB.h"
#include "utils.h"

class OctreeNode {
private:

public:
	OctreeNode();

	void add(Vertex* vertices, int count);

	AABB boundingBox;
	bool isLeaf;

	OctreeNode* children;
	int verticesCount;
	Vertex* includedVertices;
};

