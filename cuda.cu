//outsourced function is called kernel in CUDA
// __global__ sends the kernel to device
//one can set which device to use via cudaSetDevice in the main function 
//then the function will use the device

//use nvcc -o test test.cu to compile
//sample code for list squaring
#define MAX 1024
#include <stdio.h>
#include <stdlib.h> 

float in[MAX]; //creating input list memory 
float result[MAX]; //creating result list memory 

__global__ void vecSqr(float *in, float *result)
    {
        int id = threadIdx.x; //thread id
        result[id] = in[id] * in[id];
};
//this shows how CUDA memory model works
int main() {
    for (int i = 0; i < MAX; i++) {
        in[i] = rand() % 100; //filling input list with random numbers
    };
    float *device_in, *device_result; //creating pointers to input and result lists on GPU 
    size_t bytes = MAX * sizeof(float); //calculating size of input list
    cudaMalloc(&device_in, bytes); //allocating memory for input list on GPU
    cudaMalloc(&device_result, bytes); //allocating memory for result list on GPU
    cudaMemcpy(device_in, in, bytes, cudaMemcpyHostToDevice); //copying input list from CPU to GPU
    vecSqr<<<1, MAX>>>(device_in, device_result); //calling kernel function
    cudaMemcpy(result, device_result, bytes, cudaMemcpyDeviceToHost); //copying result list from GPU to CPU
    cudaFree(device_in); //freeing memory on GPU
    cudaFree(device_result); //freeing memory on GPU
    for (int i = 0; i < MAX; i++) {
        printf("%f\n", result[i]); //printing result list
    };
    //if the objects are heap alloacted, they must be freed on CPU as well
    return 0;
}

//the thread id relates each calcalation to a gpu thread
//from the structure of gpu, the calculations to threads can be sent into a single multiprocessor   
//so the threads are grouped into blocks
//however, gpus have multiple multiprocessors
//so the blocks are grouped into grids

//using two blocks
//1024 threads per block
//1 x 2 grid of blocks
#define MAX 2048
__global__ void vecSqr(float *in, float *result)
    {
        //now id uses blocks and grids
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        //blockIdx is the index of the block in the grid
        //blockDim is the number of threads in a block
        //for the first block, blockIdx.x = 0, blockDim.x = 1024, threadIdx.x = 0, 1, 2, 3, ..., 1023
        //so, id = 0, 1, 2, 3, ..., 1023
        //for the second block, blockIdx.x = 1, blockDim.x = 1024, threadIdx.x = 0, 1, 2, 3, ..., 1023
        //so, id = 1024, 1025, 1026, 1027, ..., 2047
        result[id] = in[id] * in[id];
};

int main() {
    //same from above
    vecSqr<<<2, MAX / 2>>>(device_in, device_result);
    //same from above

}

//for used threads, they must be masked 
#define MAX 1268
__global__ void vecSqr(float *in, float *result)
    {
        int id = threadIdx.x; //thread id
        if (id < MAX) {
            result[id] = in[id] * in[id]};
};

int main() {
    //same from above
    vecSqr<<<2, 1024>>>(device_in, device_result);
    //same from above
}
//one can change the dimension of the grid and block
#define MAX_x 32
#define MAX_y 32

float in[MAX_x][MAX_y]; //creating input list memory
float result[MAX_x][MAX_y]; //creating result list memory

__global__ void vecSqr(float *in, float *result)
    {
        int id_x = blockIdx.x * blockDim.x + threadIdx.x; //using multiple dimensions
        int id_y = blockIdx.y * blockDim.y + threadIdx.y;
        result[id_x][id_y] = in[id_x][id_y] * in[id_x][id_y];
};

int main() {
    for (int i = 0; i < MAX_x; i++) {
        for (int j = 0; j < MAX_y; j++) {
            in[i][j] = rand() % 100; //filling input list with random numbers
        };
    };
    float *device_in, *device_result;
    size_t bytes = MAX_x * MAX_y * sizeof(float); //calculating size of input list
    //same memory allocation and copying from above
    dim3 dimBlock(MAX_x, MAX_y); //creating block dimension
    //this can have 3 dimensions at most 
    //also grid dimension can be created
    //dim3 dimGrid(2, 2)
    //shows how the blocks are grouped into grids
    //gridDim is the number of blocks in a grid
    //use grids, blocks and basic loops to match the calculations to the data structure and hardware
    //to achieve the best performance
    vecSqr<<<1, dimBlock>>>(device_in, device_result);
    for (int i = 0; i < MAX_x; i++) {
        for (int j = 0; j < MAX_y; j++) {
            printf("%f\n", result[i][j]); //printing result list
        };
    };
    //same freeing from above
}

//2D threads to 1D calculation mapping
#define MAX 4096

__global__ void vecSqr(float *in, float *result){   
    int id = threadIdx.x + threadIdx.y * blockDim.x + ((blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y); //calculating thread id
    result[id] = in[id] * in[id]; //printing thread id
};

int main() {
    //same from 1D above
    dim3 dimBlock(32, 32); //creating block dimension
    dim3 dimGrid(2, 2); //creating grid dimension
    vecSqr<<<dimGrid, dimBlock>>>(device_in, device_result);
    //as loing as the id is well defined, blocks in the grid will allocated to the SMs automatically
    //same from 1D above
}

//error handling
//CUDA does not have a built-in error handling
//and it is not easy to debug due to the parallel nature of the code
//so, one must build a custom error handling and use it

#ifndef CHECK_ERROR_H //this is to avoid multiple definitions
#define CHECK_ERROR_H
//returns the CUDA error string associated with the given error code
#define chkErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//this is the function that actually does the error checking
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "ChkErr: %s %s line %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
}

#endif //CHECK_ERROR_H //close the ifndef

int main() {
    int *d_a;
    chkErr(cudaMalloc(&d_a, 9000000000000000000)); //this will return an error
    int *h_a = (int *)malloc(100); 
    //use the error handling for all the cuda functions
    chkErr(cudaMemcpyHostToDevice(d_a, h_a, 100));

    //syncronous error is error happening in the host //sticky error
    //asynchronous error is error happening in the device //non-sticky error
    cudaMalloc(&d_a, 9000000000000000000); //this error is not sticky //use chkErr like above 
    //also kernel execution error is asynchronous
    vecSqr<<<1, 1024>>>(d_a, h_a); //also cannot be checked with chkErr
    //so, one must use cudaDeviceSynchronize() to check for asynchronous errors
    chkErr(cudaGetLastError()); //checks kernel launching error //sticky most of the time
    chkErr(cudaDeviceSynchronize()); //keep the host waiting until the device is done
    //also chkErr in the line above will check kernel execution error
}