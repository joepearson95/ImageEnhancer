#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations

		typedef int mytype;
		vector<int> hist(256);
		size_t histSize = hist.size() * sizeof(mytype);	 
		
		vector<int> C(hist.size());

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_cumul_hist(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_map(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer dev_project(context, CL_MEM_READ_WRITE, image_input.size());

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		/*queue.enqueueWriteBuffer(dev_hist, CL_TRUE, 0, histSize, &hist.data()[0]);	
		queue.enqueueWriteBuffer(dev_map, CL_TRUE, 0, histSize, &hist.data()[0]);
		queue.enqueueWriteBuffer(dev_project, CL_TRUE, 0, histSize, &hist.data()[0]);*/

		//4.2 Setup and execute the kernel (i.e. device code)
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

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, cl::NDRange(hist.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel4, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);

		vector<unsigned char> output(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_project, CL_TRUE, 0, output.size(), &output.data()[0]);

		//cout << output << endl;
		CImg<unsigned char> output_image(output.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
