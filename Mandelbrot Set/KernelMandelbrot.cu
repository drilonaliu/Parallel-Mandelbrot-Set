#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include "KernelMandelbrot.cuh"


__global__ void kernel(uchar4* ptr, double zoomfactor, double shiftX, double shiftY, int iterations, int width, int height) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = i + j * blockDim.x * gridDim.x;

	double aspectRatio = (1.0 * width) / (1.0 * height);
	double startIntervalX = (-2) * zoomfactor * aspectRatio;
	double endIntervalX = 1 * zoomfactor * aspectRatio;

	double startIntervalY = 1.5 * zoomfactor;
	double endIntervalY = -1.5 * zoomfactor;

	//Each pixel represents a point in the Oxy coordinate system. These points are 
	double x = (endIntervalX - startIntervalX) * i / (width * 1.0) + startIntervalX + shiftX;
	double y = (endIntervalY - startIntervalY) * j / (height * 1.0) + startIntervalY + shiftY;

	double c_x = x;
	double c_y = y;
	double zX = 0, zY = 0, a = 0, b = 0;

	int max_iteration = iterations;
	int iteration;
	double squaredSums;
	for (iteration = 0; iteration < max_iteration; iteration++) {
		double zX_squared = zX * zX;
		double zY_squared = zY * zY;
		a = zX_squared - zY_squared + c_x;
		b = 2 * zX * zY + c_y;
		zX = a;
		zY = b;
		squaredSums = zX_squared + zY_squared;
		if (squaredSums > 4) {
			break;
		}
	}

	int R = 0, G = 0, B = 0;
	if (iteration < max_iteration) {
		setRGB(squaredSums, iteration, max_iteration, R, G, B);
	}

	ptr[offset].x = R;
	ptr[offset].y = G;
	ptr[offset].z = B;
	ptr[offset].w = 0;
}

__device__ void setRGB(double squaredSums, int iteration, int max_iteration, int& R, int& G, int& B) {
	double magnitude = sqrt(squaredSums);

	int max_iter = max_iteration;
	// Calculate a smooth iteration value
	float smooth_iter = iteration + 1 - log2(log2(magnitude));
	//float smooth_iter = iteration + 1 - log(log(magnitude))/log(2.0);
	// Normalize to a value between 0 and 1
	float t = smooth_iter / max_iteration;

	//float t = iteration * 1.0f / (1.0f * max_iteration);


	// Example: use HSV to RGB conversion for a smooth gradient
	float h = t * 360.0f; // Hue (0 to 360 degrees)
	float s = 1.0f;       // Saturation (constant at 1)
	float v = 1.0f;       // Value (constant at 1)

	// Convert HSV to RGB
	int i1 = int(h / 60.0f) % 6;

	float f = h / 60.0f - i1;
	float q = 1.0f - s * f;
	float p = 1.0f - s * (1 - f);

	float r = 0;
	float g = 0;
	float b = 0.0f;

	switch (i1) {
	case 0:
		r = v;
		g = p;
		b = 1.0f - s;
		break;
	case 1:
		r = q;
		g = v;
		b = 1.0f - s;
		break;
	case 2:
		r = 1.0f - s;
		g = v;
		b = p;
		break;
	case 3:
		r = 1.0f - s;
		g = q;
		b = v;
		break;
	case 4:
		r = p;
		g = 1.0f - s;
		b = v;
		break;
	case 5:
		r = v;
		g = 1.0f - s;
		b = q;
		break;
	}
	R = (int)(r * 255);
	G = (int)(g * 255);
	B = (int)(b * 255);
}