#include <stdio.h>

__device__ void vecAdd(float *h_A, float *h_B, float *h_C, int n, int numThreads)
{
 	int i = threadIdx.x + blockDim.x*blockIdx.x;
	 while (i<n)
	{
	 	h_C[i] = h_A[i] + h_B[i];
		i += numThreads;
	}
} 


__global__ void mykernel(float *h_A, float *h_B, float *h_C, int n) 
{
  //int id = threadIdx.x + blockDim.x*blockIdx.x;
  int numThreads = gridDim.x * blockDim.x;
  vecAdd(h_A, h_B, h_C, n, numThreads);
}



int main( void ) {
        int n = 1000;
	int size = n * sizeof(float);
        float *d_A, *d_B, *d_C;
	 //Inicialize os vetores A e B
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);
	int i = 0;
	while(i<n){
		h_A[i] = 0.1f * i;
		h_B[i] = 2 * i;
		h_C[i] = 0;
		i++;
	}
	 cudaMalloc((void **) &d_A, size);
	 cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	 cudaMalloc((void **) &d_B, size);
	 cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	 cudaMalloc((void **) &d_C, size); 
	mykernel<<<2,32>>>(d_A, d_B, d_C, n);
  	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	 i = 0;
	while(i<n){
		printf("\n %f",h_C[i]);
		i++;
	}
	 printf("\n");
	 cudaFree(d_A);
	 cudaFree(d_B);
	 cudaFree (d_C);
	return 0;
}
