#include <iostream>
#include <omp.h>


int main()
{
	setlocale(LC_ALL, "rus");
	int numberOfThreads = 2;  // количество потоков

#pragma omp parallel num_threads(numberOfThreads)  
	{
		// участок кода, выполняемый параллельно несколькими потоками
		// в зависимости от переменной numberOfThreads
		std::cout << "Hello world! - поток ";
		std::cout << omp_get_thread_num() << std::endl;
	}
}

