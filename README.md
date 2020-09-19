# CUDA Hot Cores Test
This project is a test to implement a concept called hot cores. Meaning that we want to keep the cuda cores hot and running a kernel all the time, doing nothing but waiting for the input to arrive, once it arrives the core runs the computation and goes back to wait for another input. This idea is meant to keep cuda cores working without re-running them to achieve better performance. A bonus is if we can make these hot cores be dynamic and be able to call different device functions as input too.

# Barriers To Implementation
- *Thread execution timeout* exists in any OS to kill kernels that run too much time (that might contain an infinite loop), the only workaround to this is to change the execution timeout of CUDA cores in the OS, which may be unsafe.
- *A thread can only wait for other threads in the same block*, to synchronize multiple block we would have to use the `cudaDeviceSynchronize()` function in the CPU and not GPU.
- *There is no instruction to wait on data or flag changes* (accept for synchronization barriers that wait on threads themselves)
- Synchronizing via global memory might take more time them re-uploading kernel code on demand.

# Implementation High Overview
While keeping in mind the implementation barriers, the implementation will be on a CUDA neural network implementation in C++ which was made by [pwlnk](https://github.com/pwlnk) in his [repository](https://github.com/pwlnk/cuda-neural-network). The implementation will be configured to use the concept of CUDA hot cores, and tested to see if any speedup was made. Still no clear idea of how the implementation should be made except for the barriers mentioned above.
