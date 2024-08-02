#include "MandelbrotRendering.cuh";
#include "KernelMandelbrot.cuh"
#include "MandelbrotVariables.cuh";
#include "CPUMandelbrot.cuh";



#include <algorithm>
#include <chrono>
#include <iostream>
#include<vector>
using namespace std;
using namespace std::chrono;



bool generatePoints = true;
double deltaX = 0;
double deltaY = 0;
float zoom = 1;
int iterations = 100;
bool CPUImplementation = false;
bool GPUImplementation = true;;

void draw_func() {
	uchar4* devPtr;
	size_t size;

	int width = glutGet(GLUT_WINDOW_WIDTH);
	int height = glutGet(GLUT_WINDOW_HEIGHT);
	dim3 grids(width / 16, height/ 16);
	dim3 threads(16, 16);
	if (GPUImplementation) {		
		cudaGraphicsMapResources(1, &resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

		kernel << <grids, threads >> > (devPtr, zoom, deltaX, deltaY, iterations, width, height);
		cudaDeviceSynchronize();

		cudaGraphicsUnmapResources(1, &resource, NULL);
		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glutSwapBuffers();
	}
	else if(CPUImplementation){
		renderMandelbrotCPU();
	}

	//cudaDeviceSynchronize()
}