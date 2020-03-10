#include <iostream>
#include <vector>
#include "CImg.h"
#include "Utils.h"

using namespace cimg_library;
using namespace std;
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	// Handle the command line options such as device selection, etc. and select the given image
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_large.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	// Run a try/catch just incase any errors arrise in the code
	try {
		// Use CImg to get the specified input image
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		// Select the computing device to use before displaying the device that is running
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << "\n\n" << std::endl;

		// Queue for pushing the commands to device - profiling is enabled to show the execution time, etc.
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load/build the sources and link to the kernel required
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error & err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Program logic for image enhancement 

		// Histogram(s) vector
		typedef int mytype;
		vector<int> hist(256);
		size_t histSize = hist.size() * sizeof(mytype);

		//Buffer creation
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_cumul_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_map(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_project(context, CL_MEM_READ_WRITE, image_input.size());

		// Copy image to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		// Create the specific kernels for each step
		cl::Kernel kernel = cl::Kernel(program, "hist_simple");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_hist);

		cl::Kernel kernel2 = cl::Kernel(program, "cumul_hist");
		kernel2.setArg(0, dev_hist);
		kernel2.setArg(1, dev_cumul_hist);

		cl::Kernel kernel3 = cl::Kernel(program, "map");
		kernel3.setArg(0, dev_cumul_hist);
		kernel3.setArg(1, dev_map);

		cl::Kernel kernel4 = cl::Kernel(program, "project");
		kernel4.setArg(0, dev_map);
		kernel4.setArg(1, dev_image_input);
		kernel4.setArg(2, dev_project);

		// Event Variables
		cl::Event prof_event;
		cl::Event prof_event2;
		cl::Event prof_event3;
		cl::Event prof_event4;

		// Enqueue command to execute on specific kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &prof_event2);
		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &prof_event3);
		queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);

		// Create output image buffer and read to it
		vector<unsigned char> output(image_input.size());
		queue.enqueueReadBuffer(dev_project, CL_TRUE, 0, output.size(), &output.data()[0]);

		// Get full info (memory transfers) for each kernel
		cout << "Kernel Full Info..." << endl;
		cout << "Kernel1: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		cout << "Kernel2: " << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << endl;
		cout << "Kernel3: " << GetFullProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << endl;
		cout << "Kernel4: " << GetFullProfilingInfo(prof_event4, ProfilingResolution::PROF_US) << "\n\n" << endl;

		// Get execution times for each kernel
		cout << "Kernel Execution Time... " << endl;
		cout << "Kernel1 execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Kernel2 execution time [ns]:" << prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Kernel3 execution time [ns]:" << prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Kernel4 execution time [ns]:" << prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		// Display output image
		CImg<unsigned char> output_image(output.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error & err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException & err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
