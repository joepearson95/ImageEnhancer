#include <iostream>
#include <vector>
#include "CImg.h"
#include "Utils.h"

using namespace cimg_library;
using namespace std;

int main(int argc, char** argv) {
	cimg::exception_mode(0);

	// Run a try/catch just incase any errors arrise in the code
	try {
		// Selection of the file and platform to run program on
		cout << "Which image would you like to use: \n[0] test.pgm \n[1] test_large.pgm\n[2] test.ppm\n[3] test_large.ppm\n[4] 16 Bit.ppm\n[5] 16 Bit.pgm\n" << endl;
		string image_filename;
		cin >> image_filename;
		if (image_filename == "0") {
			image_filename = "test.pgm";
		}
		else if (image_filename == "1") {
			image_filename = "test_large.pgm";
		}
		else if (image_filename == "2") {
			image_filename = "test.ppm";
		}
		else if (image_filename == "3") {
			image_filename = "test_large.ppm";
		}
		else if (image_filename == "4") {
			image_filename = "16bit.ppm";
		}
		else if (image_filename == "5") {
			image_filename = "16bit.pgm";
		}
		else {
			image_filename = "none";
			cout << "No file found." << endl;
		}

		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::Platform::get(&platforms);
		for (int i = 0; i < (int)platforms.size(); i++)
			cout << "Platform [" << i << "] - " << GetPlatformName(i) << "\n" << endl;

		cout << "Which platform would you like to use:" << endl;
		string platform;
		cin >> platform;

		if (platform != "0" && platform != "1" && image_filename != "none") {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}
		cout << "What bin size?" << endl;
		int histSizeNum;
		cin >> histSizeNum;
			
		cout << "Runing on " << GetPlatformName(stoi(platform)) << ", " << GetDeviceName(stoi(platform), 0) << "\n\n" << std::endl;
		// Use CImg to get the specified input image
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		// Select the computing device to use before displaying the device that is running
		cl::Context context = GetContext(stoi(platform), 0);		

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
		vector<int> hist(histSizeNum);
		size_t histSize = hist.size() * sizeof(mytype);

		//Buffer creation
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_cumul_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_map(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_project(context, CL_MEM_READ_WRITE, image_input.size());

		// Copy image to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);

		cout << "Would you like to use local or global memory for the histogram?\n [0] - Local\n [1] - Global\n" << endl;
		string memoryUsage;
		cin >> memoryUsage;

		if (memoryUsage == "0") {
			memoryUsage = "local_hist_simple";
		}
		else if (memoryUsage == "1") {
			memoryUsage = "hist_simple";
		}
		else {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}
		// Create the specific kernels for each step
		cl::Kernel kernel = cl::Kernel(program, memoryUsage.c_str());
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_hist);
		if (memoryUsage == "local_hist_simple") {
			kernel.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
			kernel.setArg(3, histSizeNum);
		}

		// Select the type of calculation required for the second kernel
		cout << "Which type of calculation:\n [0] - Scan\n [1] - Cumulative Histogram\n [2] - Local Scan\n [3] - Local Cumulative Histogram\n" << endl;
		string secondKernel;
		cin >> secondKernel;
		if (secondKernel == "0") {
			secondKernel = "scan_hist";
		}
		else if (secondKernel == "1") {
			secondKernel = "cumul_hist";
		}
		else if (secondKernel == "2") {
			secondKernel = "local_scan_hist";
		}
		else if (secondKernel == "3") {
			secondKernel = "local_cumul_hist";
		}
		else {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}

		cl::Kernel kernel2 = cl::Kernel(program, secondKernel.c_str());
		kernel2.setArg(0, dev_hist);
		kernel2.setArg(1, dev_cumul_hist);
		if (secondKernel == "local_cumul_hist") {
			kernel2.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
		}
		if (secondKernel == "local_scan_hist") {
			kernel2.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
			kernel2.setArg(3, cl::Local(histSizeNum * sizeof(mytype)));
		}

		cout << "Would you like to use local or global memory mapping?\n [0] - Local\n [1] - Global\n" << endl;
		string mapMemoryType;
		cin >> mapMemoryType;

		if (mapMemoryType == "0") {
			mapMemoryType = "local_map";
		}
		else if (mapMemoryType == "1") {
			mapMemoryType = "map";
		}
		else {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}

		cl::Kernel kernel3 = cl::Kernel(program, mapMemoryType.c_str());
		
		kernel3.setArg(0, dev_cumul_hist);
		kernel3.setArg(1, dev_map);
		if(mapMemoryType == "local_map")
			kernel3.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
		

		cl::Kernel kernel4 = cl::Kernel(program, "project");
		kernel4.setArg(0, dev_map);
		kernel4.setArg(1, dev_image_input);
		kernel4.setArg(2, dev_project);

		// Event variables to show the timeframes, etc.
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
