kernel void hist_simple(global const uchar* A, global int* H) {
	// Define the id and get the bin index
	int id = get_global_id(0);
	int bin_index = A[id];
	
	// Gather the histogram
	atomic_inc(&H[bin_index]);
}

kernel void cumul_hist(global int* A, global int* B) {
	// Define the id and size of input to loop
	int id = get_global_id(0);
	int N = get_global_size(0);

	// Loop the size and then pass to the output the cumulative histogram that is calculated
	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

kernel void scan_hist(global int* A, global int* B) {
	// Gather the size and id
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	// Loop the size and begin a simple stride
	for (int stride = 1; stride < N; stride *= 2) {
		// Pass the original input to the output based on if the current iteration is greater than the current stride
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		// Ensure that it is synced
		barrier(CLK_GLOBAL_MEM_FENCE);

		// Swap A & B between each step
		C = A; A = B; B = C;
	}
}

kernel void local_hist_simple(global const uchar* A, global int* H, local int* LH, int nr_bins) {
	// Define the id for global and local, etc.
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	
	LH[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[LH[lid]]);
	/*int bin_index = A[id];

	if (lid < nr_bins)
		LH[lid] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(&LH[bin_index]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if(id < nr_bins)
		atomic_add(&H[id], LH[lid]);*/
}

kernel void local_cumul_hist(global int* A, global int* B, local int* LH) {
	// Get variables to use
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Cache all values from global to local memory
	LH[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop over and create the cumulative histogram
	for (int i = lid + 1; i < N; i++)
		atomic_add(&B[i], LH[lid]);
}

kernel void local_scan_hist(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int* scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

kernel void local_map(global int* A, global int* B, local int* localMem) {
	// Get the id and size before scaling the values to ensure that the maximum is 255
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	localMem[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	B[id] = (double)localMem[lid] * 255 / localMem[N - 1];
}

kernel void map(global int* A, global int* B) {
	// Get the id and size before scaling the values to ensure that the maximum is 255
	int id = get_global_id(0);
	int N = get_global_size(0);
	B[id] = (double)A[id] * 255 / A[N-1];
}

kernel void project(global int* A, global uchar* image, global uchar* output) {
	// Simply get the input and project it to the output for displaying the image
	int id = get_global_id(0);
	output[id] = A[image[id]];
}