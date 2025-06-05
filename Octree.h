#pragma once

#include "OctreeNode.h"


class Octree {
private:


public:
	Octree();

	void init(Vertex* allVertices, int verticesCount);
	void build(OctreeNode* currentNode);
	
	OctreeNode* root;
};

