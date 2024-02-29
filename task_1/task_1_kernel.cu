#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>


__global__ void helloWorld() 
{
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Hello, World! from thread %d \n", threadId);
    // std::cout <<  "Hello, World! из потока " << threadId << std::endl;  не работает с CUDA, надо использовать printf
}

int main() 
{
    // setlocale(LC_ALL, "rus");  не работает с CUDA
    int numBlocks = 2;  // количество блоков
    int threadsPerBlock = 3;  // потоков в блоке

    // вычисляем общее количество потоков
    int totalThreads = numBlocks * threadsPerBlock;

    // вызываем ядро CUDA с заданным количеством блоков и потоков
    helloWorld <<<numBlocks, threadsPerBlock>>>();

    // ожидание завершения выполнения всех потоков
    cudaDeviceSynchronize();

    return 0;
}