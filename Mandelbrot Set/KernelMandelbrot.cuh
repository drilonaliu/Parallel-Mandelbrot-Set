#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

__global__ void kernel(uchar4* ptr, double zoomfactor, double shiftX, double shiftY, int iterations, int width, int height);
__device__ void setRGB(double squaredSums, int iteration, int max_iteration, int& R, int& G, int& B);