#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>


int main()
{
    setlocale(LC_ALL, "rus");

    int arraySize = 1000000;  // размер массива
    
    // генерация массива с произвольными элементами
    std::srand(std::time(nullptr));
    int* arr = new int[arraySize];
    unsigned int rangeOfNumbers = 100;  // максимальная граница элементов в массиве

    for (size_t i = 0; i < arraySize; ++i)
    {
        arr[i] = std::rand() % rangeOfNumbers;
    }
    // вывод элементов массива
    /*for (size_t i = 0; i < arraySize; ++i)
    {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;*/

    int result = 0;  // итоговая сумма
    auto start_time = std::chrono::high_resolution_clock::now(); // Время начала отсчета

    // Запуск цикла, который вычисляет сумму параллельно. 
    // reduction(+:result) означает, что значение переменной result будет зависеть от всех потоков, т.е.
    // итоговая сумма будет "собираться" из промежуточных сумм всех потоков 
#pragma omp parallel for reduction(+:result)  
    for (int i = 0; i < arraySize; ++i)
    {
        result += arr[i];
    }

    // время конца отсчета
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Итоговая сумма: " << result << std::endl;
    std::cout << "Затраченное время: " << duration.count() << " мс." << std::endl;
    std::cout << "Количество потоков: " << omp_get_max_threads() << std::endl;
    delete[] arr;
}
