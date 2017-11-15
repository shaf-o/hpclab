#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//Thread block size
#define BLOCK_SIZE 3
#define WA 3
// Matrix A width
#define HA 3
// Matrix A height
#define WB 3// Matrix B
#define HB WA
// Matrix B
#define WC WB
// Matrix C
#define HC HA
// Matrix C

//Allocates a matrix with random float entries.
void randomInit(float * data ,int size)
{
for(int i = 0; i < size; ++i)
//data[i] = rand() / (float) RAND_MAX;
data[i] = i;
}
// CUDA Kernel
__global__ void matrixMul(float* C,float* A,float* B,int wA,int wB)
{
// 2D Thread ID
int tx = threadIdx.x;
int ty = threadIdx.y;
// value stores the element that is computed by the thread
float value = 0;
for(int i = 0; i < wA; ++i)
{
float elementA = A[ty * wA + i];
float elementB = B[i * wB + tx];
value += elementA * elementB;
}
// Write the matrix to device memory each
// thread writes one element
C[ty * wA + tx] = value;
}
// Program main
int main(int argc ,char** argv)
{
// set seed for rand()
srand(2006);
// 1. allocate host memory for matrices A and B
unsigned int size_A = WA * HA;
unsigned int mem_size_A =sizeof(float) * size_A;
float* h_A = (float*) malloc(mem_size_A);
unsigned int size_B = WB * HB;
unsigned int mem_size_B =sizeof(float) * size_B;
float * h_B = (float*) malloc(mem_size_B);
// 2. initialize host memory
randomInit(h_A, size_A);
randomInit(h_B, size_B);// 3. print out A and B
printf("\n\nMatrix A\n");
for(int i = 0; i < size_A; i++)
{
printf("%f ", h_A[i]);
if(((i + 1) % WA) == 0)
printf("\n");
}
printf("\n\nMatrix B\n");
for(int i = 0; i < size_B; i++)
{
printf
("%f ", h_B[i]);
if(((i + 1) % WB) == 0)
printf("\n");
}
// 4. allocate host memory for the result C
unsigned int size_C = WC * HC;
unsigned int mem_size_C =sizeof(float) * size_C;
float * h_C = (float *) malloc(mem_size_C);
// 8. allocate device memory
float* d_A;
float* d_B;
cudaMalloc((void**) &d_A, mem_size_A);
cudaMalloc((void**) &d_B, mem_size_B);
//9. copy host memory to device
cudaMemcpy(d_A, h_A,mem_size_A ,cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B,mem_size_B ,cudaMemcpyHostToDevice);
// 10. allocate device memory for the result
float* d_C;
cudaMalloc((void**) &d_C, mem_size_C);
// 5. perform the calculation
//
//setup execution parameters
dim3 threads(BLOCK_SIZE , BLOCK_SIZE);
dim3 grid(WC / threads.x, HC / threads.y);
//
//execute the kernel
matrixMul<<< grid , threads >>>(d_C, d_A,d_B, WA, WB);
// 11. copy result from device to host
cudaMemcpy(h_C, d_C, mem_size_C ,cudaMemcpyDeviceToHost);
// 6. print out the results
printf("\n\n Matrix C ( Results ) \n ");
for(int i = 0;i<size_C; i ++){
printf("%f",h_C[i]);
if(((i+ 1) % WC) == 0)
printf("\n");
}printf("\n");
// 7.clean up memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
free(h_A);
free(h_B);
free(h_C);
}

