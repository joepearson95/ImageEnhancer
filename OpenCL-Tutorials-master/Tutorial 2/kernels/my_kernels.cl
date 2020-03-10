kernel void hist_simple(global const uchar* A, global int* H) {
	int id = get_global_id(0);

	int bin_index = A[id];
	atomic_inc(&H[bin_index]);
}

kernel void cumul_hist(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

kernel void map(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	B[id] = A[id] * 255 / A[N-1];
}

kernel void project(global int* A, global uchar* image, global uchar* output) {
	int id = get_global_id(0);

	output[id] = A[image[id]];
}