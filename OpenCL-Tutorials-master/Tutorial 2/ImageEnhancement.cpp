#include <iostream>
#include <vector>
#include <fstream>
#include "CImg.h"
#include "Utils.h"

using namespace cimg_library;
using namespace std;

int main(int argc, char** argv) {
	cimg::exception_mode(0);

	// Run a try/catch just incase any errors arrise in the code
	try {
		// User input section
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

		// Loop the platforms and display them for the user to choose
		vector<cl::Platform> platforms;
		vector<cl::Device> devices;
		cl::Platform::get(&platforms);
		for (int i = 0; i < (int)platforms.size(); i++)
			cout << "Platform [" << i << "] - " << GetPlatformName(i) << "\n" << endl;

		cout << "Which platform would you like to use:\n" << endl;
		string platform;
		cin >> platform;

		if (platform != "0" && platform != "1" && image_filename != "none") {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}

		// Define (from user input) what the the bin size is for this program
		cout << "What bin size?\n" << endl;
		int histSizeNum;
		cin >> histSizeNum;
		
		// Define (based on user input) what the program will use
		cout << "Would you like to use local or global memory for the image?\n [0] - Local\n [1] - Global\n" << endl;
		string memoryType;
		cin >> memoryType;

		// Histogram Memory Definition
		string memoryUsage;
		if (memoryType == "0") {
			memoryUsage = "local_hist_simple";
		}
		else if (memoryType == "1") {
			memoryUsage = "hist_simple";
		}
		else {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}

		// Map Definition Based On Memory Choice
		string mapMemoryType;
		if (memoryType == "0") {
			mapMemoryType = "local_map";
		}
		else if (memoryType == "1") {
			if (image_filename == "16bit.ppm" || image_filename == "16bit.pgm") {
				mapMemoryType = "sixteen_bit_map";
			}
			else {
				mapMemoryType = "map";
			}
		}
		else {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}

		// Select The Type of Calculation Required For The Second Kernel
		cout << "Which type of calculation:\n [0] - Scan\n [1] - Cumulative Histogram\n" << endl;
		string secondKernel;
		cin >> secondKernel;

		if (memoryType == "1" && secondKernel == "0") {
			secondKernel = "scan_hist";
		}
		else if (memoryType == "1" && secondKernel == "1") {
			if (image_filename == "16bit.ppm" || image_filename == "16bit.pgm") {
				secondKernel = "sixteenbit_cumul_hist";
			} 
			else {
				secondKernel = "cumul_hist";
			}
		}
		else if (memoryType == "0" && secondKernel == "0") {
			secondKernel = "local_scan_hist";
		}
		else if (memoryType == "0" && secondKernel == "1") {
			secondKernel = "local_cumul_hist";
		}
		else {
			cout << "Cannot run program with the given inputs. Now exiting...";
			exit(0);
		}

		// Display output of certain selections
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

		typedef short 16bitImage;
		typedef int normalImage;
		vector<mytype> hist(histSizeNum);
		size_t histSize = hist.size() * sizeof(mytype);		

		//Buffer creation
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_cumul_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_map(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_project(context, CL_MEM_READ_WRITE, image_input.size());

		// Copy image to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(dev_hist, 0, 0, histSize);

		// Kernel 1 is used for the creation of a histogram - local or global is defined form user input at the start of the program run
		cl::Kernel kernel = cl::Kernel(program, memoryUsage.c_str());
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, dev_hist);
		if (memoryUsage == "local_hist_simple") {
			kernel.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
			kernel.setArg(3, histSizeNum);
		}

		// Kernel 2 is used computing partial reductions with a scan or a cumulative histogram - also memory is based off of the beginning user input 
		cl::Kernel kernel2 = cl::Kernel(program, secondKernel.c_str());
		if (secondKernel == "scan_hist") {
			kernel2.setArg(0, dev_hist);
		}
		if (secondKernel == "cumul_hist" || secondKernel == "sixteenbit_cumul_hist" || secondKernel == "local_cumul_hist" || secondKernel == "local_scan_hist") {
			kernel2.setArg(0, dev_hist);
			kernel2.setArg(1, dev_cumul_hist);
		}
		if (secondKernel == "local_cumul_hist") {
			kernel2.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
		}
		if (secondKernel == "local_scan_hist") {
			kernel2.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));
			kernel2.setArg(3, cl::Local(histSizeNum * sizeof(mytype)));
		}

		// Kernel 3 is for creating a LUT for the final kernel when creating the image again
		cl::Kernel kernel3 = cl::Kernel(program, mapMemoryType.c_str());
		if (secondKernel == "scan_hist") {
			kernel3.setArg(0, dev_hist);
		}
		else {
			kernel3.setArg(0, dev_cumul_hist);
		}
		kernel3.setArg(1, dev_map);
		if (mapMemoryType == "local_map")
			kernel3.setArg(2, cl::Local(histSizeNum * sizeof(mytype)));

		// Kernel 4 is for projecting the data back to the image in order to perform contrast changes
		cl::Kernel kernel4 = cl::Kernel(program, "project");
		kernel4.setArg(0, dev_map);
		kernel4.setArg(1, dev_image_input);
		kernel4.setArg(2, dev_project);

		// Event variables to show the timeframes, etc.
		cl::Event prof_event;
		cl::Event prof_event2;
		cl::Event prof_event3;
		cl::Event prof_event4;

		std::ofstream txthist;
		// Enqueue command to execute on specific kernels
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &prof_event2);
		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange, NULL, &prof_event3);
		queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);


		// Create output image buffer and read to it
		vector<unsigned char> output(image_input.size());
		queue.enqueueReadBuffer(dev_project, CL_TRUE, 0, output.size(), &output.data()[0]);
		cout << output.data()[0];
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
