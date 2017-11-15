#include <stdio.h>
/**********************
* using local memory *
**********************/
// a __device__
__global__ void use_local_memory_GPU(float in)

{
float f;
f = in;
// variable "f" is in local memory and private to each
// parameter "in" is in local memory and private to each// ... real code would presumably do other stuff here ...
}
/**********************
* using global memory *
**********************/
// a __global__ function runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float *array)
{
// "array" is a pointer into global memory on the device
array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}
/**********************
* using shared memory *
**********************/
// (for clarity, hardcoding 128 threads/elements and omitting out-of-
__global__ void use_shared_memory_GPU(float *array)
{
// local variables, private to each thread
int i, index = threadIdx.x;
float average, sum = 0.0f;
// __shared__ variables are visible to all threads in the thread
// and have the same lifetime as the thread block
__shared__ float sh_arr[128];
// copy data from "array" in global memory to sh_arr in shared
// here, each thread is responsible for copying a single element.
sh_arr[index] = array[index];
__syncthreads();
// ensure all the writes to shared memory have
// now, sh_arr is fully populated. Let's find the average of all
for (i=0; i<index; i++) { sum += sh_arr[i]; }
average = sum / (index + 1.0f);
printf("Thread id = %d\t Average = %f\n",index,average);
// if array[index] is greater than the average of array[0..index-1],
// since array[] is in global memory, this change will be seen by the
// other thread blocks, if any)
if (array[index] > average) { array[index] = average; }
// the following code has NO EFFECT: it modifies shared memory, but
// the resulting modified data is never copied back to global memory// and vanishes when the thread block completes
sh_arr[index] = 3.14;
}
int main(int argc, char **argv)
{
/*
* First, call a kernel that shows using local memory
*/
use_local_memory_GPU<<<1, 128>>>(2.0f);
/*
* Next, call a kernel that shows using global memory
*/
float h_arr[128];
// convention: h_ variables live on host
float *d_arr;
// convention: d_ variables live on device (GPU
// allocate global memory on the device, place result in "d_arr"
cudaMalloc((void **) &d_arr, sizeof(float) * 128);
// now copy data from host memory "h_arr" to device memory "d_arr"
cudaMemcpy((void *)d_arr, (void *)h_arr, sizeof(float) * 128,cudaMemcpyHostToDevice);
// launch the kernel (1 block of 128 threads)
use_global_memory_GPU<<<1, 128>>>(d_arr); // modifies the contents
// copy the modified array back to the host, overwriting contents ofh_arr
cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128,cudaMemcpyDeviceToHost);
// ... do other stuff ...
/*
* Next, call a kernel that shows using shared memory
*/
// as before, pass in a pointer to data in global memory
use_shared_memory_GPU<<<1, 128>>>(d_arr);
// copy the modified array back to the host
cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128,cudaMemcpyHostToDevice);
// ... do other stuff ...
// force the printf()s to flush
cudaDeviceSynchronize();
return 0;
}
