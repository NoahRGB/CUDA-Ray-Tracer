#pragma once

#include "OctreeNode.h"

class Queue {
private:


public:
    int capacity;
    int used;

    OctreeNode* data;

    __host__ __device__ Queue() {
        capacity = 10;
        used = 0;
        data = new OctreeNode[capacity];
    }

    __host__ __device__ void enqueue(OctreeNode item) {
        if (used < capacity) {
            data[used] = item;
            used++;
        }
        else {
            OctreeNode* newData = new OctreeNode[capacity * 2];
            for (int i = 0; i < capacity; i++) {
                newData[i] = data[i];
            }
            capacity *= 2;
            delete data;
            data = newData;
            enqueue(item);
        }
    }

    __host__ __device__ OctreeNode dequeue() {
        OctreeNode itemToRemove = data[0];

        for (int i = 1; i < used; i++) {
            data[i - 1] = data[i];
        }

        used--;

        return itemToRemove;
    }

    __host__ __device__ void print() {
        //std::cout << "Queue with capacity " << capacity << " and " << used << " used spaces" << std::endl;
        //for (int i = 0; i < capacity; i++) {
        //    std::cout << data[i] << ", ";
        //}
        //std::cout << std::endl << std::endl << std::endl;
    }

};