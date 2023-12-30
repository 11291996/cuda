//error handling

#include <stdio.h>
#include <stdlib.h> 

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
}