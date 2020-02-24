// Reference: Parts adapted from https://github.com/gcielniak/OpenCL-Tutorials
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include "CImg.h"
#include <fstream>
#include <iostream>

int main() {
	std::string imageName = "test.ppm"; // Filename of the image to be used {default is test}
	using namespace cimg_library; // Namespace of cimg library for ease of use
	 // Obtain the actual image data
	CImg<unsigned char> imageData(imageName.c_str());
	CImgDisplay displayImg(imageData, "input");  
	
	// Creation of the platform (vendor specific openCL implementation)
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);	
	auto platform = platforms.front();

	// Get the Physical device to use
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	auto device = devices.front();
	
	// Read in the kernel file for usage
	std::ifstream readInFile("mykernel.cl");
	std::string src(std::istreambuf_iterator<char>(readInFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	// Get the devices that are going to work with one another
	cl::Context context(device);
	cl::Program program(context, sources);
	auto err = program.build("-cl-std=CL1.2");

	// Create the buffers for the in and output of the image.
	cl::Buffer imgIn(context, CL_MEM_READ_ONLY, imageData.size());
	cl::Buffer imgOut(context, CL_MEM_READ_WRITE, imageData.size());

	// Create a queue for sending commands to the device
	cl::CommandQueue queue(context, device);
	queue.enqueueWriteBuffer(imgIn, CL_TRUE, 0, imageData.size(), &imageData.data()[0]);

	// Create the kernel (the blueprint for how the program will work)
	cl::Kernel kernel = cl::Kernel(program, "identity");
	// Pass the arguments (the image) to the kernel
	kernel.setArg(0, imgIn);
	kernel.setArg(1, imgOut);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(imageData.size()), cl::NullRange);
	std::vector<unsigned char> output_buffer(imageData.size());
	queue.enqueueReadBuffer(imgIn, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
	
	// Begin the output of the image
	CImg<unsigned char> imgOutput(output_buffer.data(), imageData.width(), imageData.height(), imageData.depth(), imageData.spectrum());
	CImgDisplay outputDisplay(imgOutput, "output");
	
	// Keep the image open
	while (!displayImg.is_closed() && !outputDisplay.is_closed() && !displayImg.is_keyESC() && !outputDisplay.is_keyESC()) {
		displayImg.wait(1);
		outputDisplay.wait(1);
	}
}