__kernel void mykernel(__global char* data) {
	data[0] = 'H';
	data[1] = 'e';
	data[2] = 'l';
	data[3] = 'l';
	data[4] = 'o';
	data[5] = 'w';
	data[6] = 'o';
	data[7] = 'r';
	data[8] = 'l';
	data[9] = 'd';
}