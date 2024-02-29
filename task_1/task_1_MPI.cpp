#include <iostream>
#include "mpi.h"
#include <cstdio>

int main(int* argc, char* argv[])
{
    // setlocale(LC_ALL, "rus"); проблемы при выводе в консоли
    int numThreads, rank;  // количество потоков и ранг данного потока
    MPI_Init(argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numThreads);

    std::cout << "Hello world! Current number of threads is " << numThreads << ", rank: " << rank << std::endl;

    MPI_Finalize();
    // запуск программы через терминал командой:
    // mpiexec .\task_1_MPI.exe, заранее перейдя в директорию проекта
}
