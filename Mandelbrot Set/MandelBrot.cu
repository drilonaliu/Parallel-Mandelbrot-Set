// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <device_launch_parameters.h>
//MandelbrotSet includes
#include "MandelbrotSet.cuh";
#include "KernelMandelbrot.cuh"
#include "MandelbrotRendering.cuh";
#include "MandelbrotVariables.cuh";
#include "UserInteraction.cuh";

using namespace std;

#define DIM 512 ;
GLuint bufferObj;
cudaGraphicsResource* resource;

int dim = 512;
float zoomFactor = 1.0f;
double x = 1.0;
float goLeft = 0;
float goRight = 0;
float goUp = 0;
float goDown = 0;

void bindFunctionsToWindow();
void initializeWindow(int argc, char** argv);
void setUpCudaOpenGLInterop();


void startMandelbrotSet(int argc, char** argv) {
	initializeWindow(argc, argv);
	setUpCudaOpenGLInterop();
	bindFunctionsToWindow();
	createMenu();
	printControls();
	glutMainLoop();
}

void initializeWindow(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Mandelbrot Set");
	glewInit();
}

void bindFunctionsToWindow() {
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
	glutSpecialFunc(specialKeyHandler);
	glutMouseWheelFunc(mouseWheel);
}

void setUpCudaOpenGLInterop() {
	//Choose the most suitable CUDA device based on the specified properties (in prop). It assigns the device ID to dev.
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaError_t error = cudaChooseDevice(&dev, &prop);
	if (error != cudaSuccess) {
		printf("Error choosing CUDA device: %s\n", cudaGetErrorString(error));
	}
	cudaGLSetGLDevice(dev);

	//Generate openGL buffer
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 2000 * 2000* 4, NULL, GL_DYNAMIC_DRAW_ARB);

	//Notify CUDA runtime that we intend to share the OpenGL buffer named bufferObj with CUDA.//FlagsNone, ReadOnly, WriteOnly
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}
