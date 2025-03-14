#include <stdio.h>
#include "matrix_mul.h"
#include <time.h>

// Thread block size
#define BLOCK_SIZE 16 

// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, int, float*);
//__global__ void Muld(float*, float*, int, int, float*);

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B


void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	int size;
	clock_t inicia;
	float tiempo;
	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	printf("Tamaño en bytes: %d \n",size);
	cudaMalloc((void**)&Ad, size);
	inicia=clock();
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	tiempo=(clock()-inicia)/(double) CLOCKS_PER_SEC;
	printf("Tiempo copia A:%f \n", tiempo);
	printf("Ancho de banda transf A: %f\n",(size/tiempo)/1000000);
	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
        inicia=clock();
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
        tiempo=(clock()-inicia)/(double) CLOCKS_PER_SEC;
        printf("Tiempo copia B:%f \n", tiempo);
        printf("Ancho de banda transf B: %f\n",(size/tiempo)/1000000);



	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((wB / dimBlock.x)+1, (hA / dimBlock.y)+1);

	// Launch the device computation
    inicia=clock();
	Muld<<<dimGrid,dimBlock>>>(Ad, Bd, hA, wB, wA, Cd);
	//Muld<<<dimGrid,dimBlock>>>(Ad, Bd, wA, wB, Cd);
	cudaDeviceSynchronize();
        tiempo=(clock()-inicia)/(double) CLOCKS_PER_SEC;
        printf("Tiempo mul externo:%f \n", tiempo);
	printf("Rendimiento Kernel: %f\n",((2*(double)hA*wA*wB)/tiempo)/1000000);

	// Read C from the device
        inicia=clock();
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
        tiempo=(clock()-inicia)/(double) CLOCKS_PER_SEC;
        printf("Tiempo copia C:%f \n", tiempo);
        printf("Ancho de banda transf C: %f\n",(size/tiempo)/1000000);

	

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}
#if 0
__global__ void Muld(float* A, float* B, int hA, int wB, int wA, float* C)
{
	int i;
	float value=0.0;
	int idx=(blockDim.x*blockIdx.x+threadIdx.x);
	int idy=(blockDim.y*blockIdx.y+threadIdx.y);
	int id=(idy*wB+idx);

	//To Do
	//C[id]=0.0;
	if(idx<wB && idy<hA){
		for (i=0; i<wA; i++){
			//C[id]+=A[idy*wA+i]*B[i*wB+idx];
			value+=A[idy*wA+i]*B[i*wB+idx];
		}
		C[id]=value;
	}
	


}
#endif

//#if 0

// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
__global__ void Muld(float* A, float* B, int hA, int wB, int wA, float* C)
{

	// Block index
	int by = blockIdx.y;
	int bx = blockIdx.x;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int idx=(blockDim.x*bx+tx);
	int idy=(blockDim.y*by+ty);
	int id=(idy*wB+idx);

	// Index of the first sub-matrix of A processed by the block
	int aBegin = by*wA*BLOCK_SIZE;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = wA-1+aBegin;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// The element of the block sub-matrix that is computed
	// by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B required to
	// compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Shared memory for the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from global memory to shared memory;
		// each thread loads one element of each matrix
		As[ty][tx] = A[a+(ty*wA+tx)];
		Bs[ty][tx] = B[b+(ty*wB+tx)];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub+=As[ty][k]*Bs[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write the block sub-matrix to global memory;
	// each thread writes one element
	if(idx<wB && idy<hA){
		C[id]=Csub;
	}
}

//#endif
