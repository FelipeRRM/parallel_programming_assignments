__global__ void mykernel( void ) {
	
}

#include <stdio.h>

int main( void ) {
	mykernel<<<1,1>>>();
	printf( "Hello World!\n" );
	return 0;
}
