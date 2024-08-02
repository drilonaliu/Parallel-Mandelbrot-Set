// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>
#include "MandelbrotVariables.cuh";
#include "UserInteraction.cuh";

float zoomLevel = 1.0f;
double offsetX = 0.0, offsetY = 0.0;
double lastMouseX, lastMouseY;
bool isPanning = false;
double shiftX = 0.0;
double shiftY = 0.0;
double lastX = 0;
double lastY = 0;

enum {
	MENU_OPTION_1,
	MENU_OPTION_2,
	MENU_EXIT
};

void mouseButton(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			isPanning = true;
			lastMouseX = x;
			lastMouseY = y;
		}
		else if (state == GLUT_UP) {
			isPanning = false;
		}
	}
}

void mouseWheel(int button, int dir, int x, int y) {
	if (dir > 0) {
		zoomLevel /= 1.1f;
		// Zoom in
	}
	else {
		// Zoom out	
		zoomLevel *= 1.1f;
	}

	zoom = zoomLevel;
	//generatePoints = false;
	//updateProjection();
	glutPostRedisplay(); // Redraw the scene
}

void mouseMove(int x, int y) {
	if (isPanning) {
		// Get the window dimensions
		int imageW = glutGet(GLUT_WINDOW_WIDTH);
		int imageH = glutGet(GLUT_WINDOW_HEIGHT);

		double dx = (double)(lastMouseX-x) / imageW;
		double dy = (double)(lastMouseY-y) / imageH; // Note: Y is inverted

		// Scale the movement by the zoom level and the orthographic projection extents
		double orthoWidth = 3.0 * zoomLevel; // Assuming initial projection is -1.0 to 1.0
		double orthoHeight = 3.0 * zoomLevel; // Assuming initial projection is -1.0 to 1.0

		offsetX += dx * orthoWidth;
		offsetY += dy * orthoHeight;

		deltaX = offsetX;
		deltaY = offsetY;

		// Update last mouse position
		lastMouseX = x;
		lastMouseY = y;
		glutPostRedisplay();
	}
}

void specialKeyHandler(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_LEFT:
		printf("Left arrow key pressed at (%d, %d)\n", x, y);
		goLeft += 0.05 * zoomFactor;
		break;
	case GLUT_KEY_RIGHT:
		goRight += 0.05 * zoomFactor;
		break;
	case GLUT_KEY_UP:
		iterations += 2;
		goUp += 0.05 * zoomFactor;
		break;
	case GLUT_KEY_DOWN:
		goDown += 0.05 * zoomFactor;
		iterations -= 2;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void key_func(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		// clean up OpenGL and CUDA
		//cudaGraphicsUnregisterResource(resource);
		//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		//glDeleteBuffers(1, &bufferObj);
		//exit(0);
		break;
	}
}

void handleMenu(int option) {
	switch (option) {
	case MENU_OPTION_1:
		CPUImplementation = true;
		GPUImplementation = false;
		break;
	case MENU_OPTION_2:
		CPUImplementation = false;
		GPUImplementation = true;
		break;
	case MENU_EXIT:
		exit(0);
		break;
	}
	glutPostRedisplay();
}

void createMenu() {
	int menu = glutCreateMenu(handleMenu);
	glutAddMenuEntry("Switch to CPU implementation", MENU_OPTION_1);
	glutAddMenuEntry("Switch to GPU impplementation", MENU_OPTION_2);
	glutAddMenuEntry("Exit", MENU_EXIT);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void printControls() {
	printf("\nRight click on window to switch between CPU and GPU implementations.");
	printf("\nPress [R] to reset ");
	printf("\nPress [arrowUp] to increase the iteration");
	printf("\nPress [arrowDown] to decrease the iteration");
	printf("\nUse left mouse click button to move around");
	printf("\nUse mouse scroll wheel to zoom in");
	printf("\n\n");
}



