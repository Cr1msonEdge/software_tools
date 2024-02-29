#include <iostream>
#include "mpi.h"
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <chrono>


int main(int argc, char* argv[])
{
    std::srand(std::time(nullptr));

    const int arraySize = 1000000;
    unsigned int rangeOfNumbers = 100;  // максимальная граница элементов в массиве
    int rank, size;
    int* arr = new int[arraySize];
    long long int partial_sum = 0;
    long long int total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // инициализация массива
    for (int i = 0; i < arraySize; i++)
    {
        arr[i] = std::rand() % rangeOfNumbers;
    }

    auto start_time = std::chrono::high_resolution_clock::now(); // Время начала отсчета
    // вычисление частичной суммы
    for (int i = rank; i < arraySize; i += size) 
    {
        partial_sum += arr[i];
    }

    // собираем промежуточные суммы 
    MPI_Reduce(&partial_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // время конца отсчета
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (rank == 0)
    {
        std::cout << "Total Sum: " << total_sum << std::endl;
        std::cout << "Time spend: " << duration.count() << " miliseconds" << std::endl;
        std::cout << "Max number of ranks: " << size << std::endl;
    }

    MPI_Finalize();
    delete[] arr;
    return 0;
}
