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

#include "CPUMandelbrot.cuh";


#include <algorithm>
#include <chrono>
#include <iostream>
#include<vector>
using namespace std;
using namespace std::chrono;

void setColor2(double zX, double zY, int iteration, int max_iter);
void setupProjection();


//Zoom Factors
float zF = 1.0f; // Zoom factor
float centerX = -0.5f; // Default center on the x-axis
float centerY = 0.0f; // Default center on the y-axis


void setupProjection() {
	glMatrixMode(GL_PROJECTION); // Set the projection matrix
	glLoadIdentity(); // Reset any previous transformations

	float halfWidth = (1.5f / zF); // Adjusted width for zoom level
	float halfHeight = (1.0f / zF); // Adjusted height for zoom level

	glOrtho(centerX - halfWidth, centerX + halfWidth,
		centerY - halfHeight, centerY + halfHeight,
		-1.0f, 1.0f); // Set orthographic projection

	glMatrixMode(GL_MODELVIEW); // Switch back to modelview matrix
	glLoadIdentity(); // Reset transformations in the modelview matrix
}


void renderMandelbrotCPU(){

	setupProjection();

	// Clear the color buffer
	glClear(GL_COLOR_BUFFER_BIT);

	// Set the point size
	glPointSize(1.0f); // Set the point size to 5 pixels

	// Draw points
	glBegin(GL_POINTS);

	// Set the color for the points
	glColor3f(0.0f, 0.31f, 0.45f); // coolblue

	int startX = -2;
	int endX = 1;

	int startY = -1;
	int endY = 1;

	int scale = 1000;
	double a = 0;
	double b = 0;

	int rows = 300;
	int cols = 200;

	auto start = high_resolution_clock::now();

	// Draw some points in a grid pattern
	for (int x = -2000; x <= 1000; x++) {
		for (int y = -1000; y <= 1000; y++) {
			double zX = 0;
			double zY = 0;
			double c_x = x * 0.001;
			double c_y = y * 0.001;

			int iteration;
			for (iteration = 0; iteration < 50; iteration++) {
				a = zX * zX - zY * zY + c_x;
				b = 2 * zX * zY + c_y;
				zX = a;
				zY = b;

				if (zX * zX + zY * zY > 4) {
					break;
				}
			}

			setColor2(zX, zY, iteration, 50);
			glVertex2f(x * 0.001, y * 0.001);
			// Normalize the coordinates

		}
	}

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	cout << "\nTime taken by function: "
		<< duration.count() << " microseconds" << endl;


	glEnd();
	// Swap buffers to display the rendered content
	glutSwapBuffers();
}

void setColor2(double zX, double zY, int iteration, int max_iter) {
	// Get the magnitude of the final complex number
	double magnitude = sqrt(zX * zX + zY * zY);

	// Calculate a smooth iteration value
	float smooth_iter = iteration + 1 - log2(log2(magnitude));

	// Normalize to a value between 0 and 1
	float t = smooth_iter / max_iter;

	// Example: use HSV to RGB conversion for a smooth gradient
	float h = t * 360.0f; // Hue (0 to 360 degrees)
	float s = 1.0f;       // Saturation (constant at 1)
	float v = 1.0f;       // Value (constant at 1)

	// Convert HSV to RGB
	int i = int(h / 60.0f) % 6;
	float f = h / 60.0f - i;
	float q = 1.0f - s * f;
	float p = 1.0f - s * (1 - f);

	float r = 0;
	float g = 0;
	float b = 0.0f;
	switch (i) {
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

	glColor3f(r, g, b); // Set the calculated RGB color
}