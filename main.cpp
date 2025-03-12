
#include <iostream>

#include "Window.h"
#include "OBJ_Loader.h"

int main() {
	
	Window window(1000, 1000, "Ray Tracing");
	
	if (window.init() == 0) {
		window.run();
	}
	else {
		std::cout << "Failed to initialise window" << std::endl;
	}

	return 0;
}