#include <stdio.h>
#include <stdlib.h>

#ifndef checkCudaErrors
static void HandleError( cudaError_t err, const char *file, int line ) {
 if (err != cudaSuccess) {
 printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
 file, line );
 exit( EXIT_FAILURE );
 }
}
#define checkCudaErrors( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif

__global__ void reduce1(float *g_idata, float *g_odata, unsigned int n)
{
extern __shared__ float sdata[]; //Dynamic shared memory allocation:
 //will specify size later
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
// Load from global to shared mem
 sdata[tid] = (i < n) ? g_idata[i] : 0;
 __syncthreads();
 for(unsigned int s = 1; s < blockDim.x; s *= 2) {
int index = 2 * s * tid;
 if(index < blockDim.x) {
 sdata[index] += sdata[index + s];
 }
__syncthreads(); 
}
// Write result for this block to global mem
 if (tid == 0) g_odata[blockIdx.x] = sdata[0];
} 

int reducaoSequencial(float* vector, int size){
	int red =  0;
	int i = 0;
	while(i<size){
		red += vector[i];
		i++;
	}
	return red;
}

float* inicializaVector(int size){
	float* vec = (float *) malloc(size*sizeof(float));
	int i = 0;
	while(i<size){
		vec[i] = i*2;
		i++;
	}
	return vec;
}

void main(){
 int n = 1000;
 float *c_idata = inicializaVector(n);
 float *c_odata = (float *) malloc (n*sizeof(float));
 cudaEvent_t start, stop;
 HANDLE_ERROR( cudaEventCreate( &start ) );
 HANDLE_ERROR( cudaEventCreate( &stop ) );
 HANDLE_ERROR( cudaEventRecord( start, 0 ) );
 /* INÍCIO DA LÓGICA DA REDUÇÃO */

 int threadsX = 32;
 int size = n * sizeof(float);
 float *d_idata, *d_odata;
 cudaMalloc((void **) &d_idata, size);
 cudaMemcpy(d_idata, c_idata, size, cudaMemcpyHostToDevice);

 cudaMalloc((void **) &d_odata, size);

 const dim3 blockSize(threadsX, 1, 1);
 const dim3 gridSize( (int) ceil((double)n/(double)threadsX), 1, 1);
 int smemSize = threads * sizeof(float);
 reduce1<<<gridSize, blockSize, smemSize>>>(d_idata, d_odata, size); 

 cudaMemcpy(c_odata, d_odata, size, cudaMemcpyDeviceToHost); 

 /* TÉRMINO DA LÓGICA DA REDUÇÃO */
 HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
 HANDLE_ERROR( cudaEventSynchronize( stop ) );
 float elapsedTime;
 HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
 printf( "Time to generate: %3.1f ms\n", elapsedTime );
 HANDLE_ERROR( cudaEventDestroy( start ) );
 HANDLE_ERROR( cudaEventDestroy( stop ) );


/*


	float resultado = reducaoSequencial(vec,n);
	printf("Resultado %f\n", resultado);
	free(vec); */
}
