#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>


// CUDA ядро для вычисления суммы массива
__global__ void sumArray(int* a, long long* result, int size) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long sum = 0;

    // каждый поток вычисляет сумму своего блока данных
    while (tid < size) 
    {
        sum += a[tid];
        tid += blockDim.x * gridDim.x;
    }

    // каждый блок помещает свою сумму в разделяемую память
    __shared__ long long blockSum[256];
    blockSum[threadIdx.x] = sum;
    __syncthreads();

    // первый поток в блоке суммирует результаты
    if (threadIdx.x == 0) 
    {
        for (int i = 1; i < blockDim.x; ++i)
        {
            blockSum[0] += blockSum[i];
        }
        result[blockIdx.x] = blockSum[0];
    }
}


int main() 
{
    std::srand(std::time(nullptr));
    const int arraySize = 1000000;  // размер массива
    const int threadsPerBlock = 12;  // количество потоков в каждом блоке 
    const int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;  // количество блоков в сетке

    const int rangeOfNumbers = 100;  // максимальная граница элементов в массиве
    // инициализация массива 
    int* arr = new int [arraySize];
    for (int i = 0; i < arraySize; ++i)
    {
        arr[i] = std::rand() % rangeOfNumbers;
    }

    int* d_array;
    long long* d_result;


    // время начала отсчета
    auto startTime = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&d_array, arraySize * sizeof(int));
    cudaMalloc((void**)&d_result, numBlocks * sizeof(long long));

    cudaMemcpy(d_array, arr, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // запуск CUDA ядра
    sumArray <<<numBlocks, threadsPerBlock>>> (d_array, d_result, arraySize);

    // копирование результата обратно на хост
    long long h_result[numBlocks];
    cudaMemcpy(h_result, d_result, numBlocks * sizeof(long long), cudaMemcpyDeviceToHost);

    // суммирование промежуточных сумм от блоков
    long long result = 0;
    for (int i = 0; i < numBlocks; ++i)
    {
        result += h_result[i];
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    printf("Sum of elements: %lld\n", result);
    printf("Time spent: %lld miliseconds \n", (endTime - startTime).count());

    // Освобождение выделенной памяти
    cudaFree(d_array);
    cudaFree(d_result);
    delete[] arr;
    return 0;
}
