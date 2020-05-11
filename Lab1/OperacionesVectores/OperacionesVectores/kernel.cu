#include <stdio.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace std;

//DEVICE

__global__ void kernelVector_x_constant(float* arr, float* res, int n, int k)
{
	//Obtengo el indice del hilo fisico
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Mientras el hilo sea valido para la operacion
	if (idx < n)
	{
		//Multiplico el elemento por la constante
		res[idx] = arr[idx] * k;
	}
}

__global__ void KernelVector_add_constant(float* arr, float* res, int n, int c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
	{
		res[idx] = arr[idx] + c;
	}
}

__global__ void KernelVector_substract_constant(float* arr, float* res, int n, int c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
	{
		res[idx] = arr[idx] - c;
	}
}

__global__ void KernelVector_add_vector(float* arr, float* newArr, float* res, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
	{
		res[idx] = arr[idx] + newArr[idx];
	}
}

__global__ void KernelVector_substract_vector(float* arr, float* newArr, float* res, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
	{
		res[idx] = arr[idx] - newArr[idx];
	}
}

//HOST
int main()
{
	int size = 1000000;
	//Separo memoria en la RAM del HOST
	float* arr;
	float* newArr;
	float constant = 5;
	float* res;
	int print_res = 30;

	cudaMallocManaged(&arr, size * sizeof(float));
	cudaMallocManaged(&newArr, size * sizeof(float));
	cudaMallocManaged(&res, size * sizeof(float));

	for (int i = 0; i < size; i++) {
		arr[i] = i;
		newArr[i] = i;
	}


	///////////////////////// EJECUTO EL KERNEL DE CUDA ////////////////////////////
	//////// 512 Hilos
	//////// ceil(1000000/512) Bloques

	cout << "Vector por constante" << endl<<'[';
	kernelVector_x_constant <<< ceil(size / 512.0), 512 >>> (arr,res, size, constant);
	//Fuerzo una llamada Sincrona
	cudaThreadSynchronize();
	for (int index = 0; index < print_res; index++)
	{
		cout << res[index] << " , ";
	}
	cout << ']' << endl;
	
	cout << "Vector mas constante" << endl << '[';
	KernelVector_add_constant <<< ceil(size / 512.0), 512 >>> (arr,res,size,constant);
	//Fuerzo una llamada Sincrona
	cudaThreadSynchronize();
	for (int index = 0; index < print_res; index++)
	{
		cout << res[index] << " , ";
	}
	cout << ']' << endl;

	cout << "Vector menos constante" << endl << '[';
	KernelVector_substract_constant <<< ceil(size / 512.0), 512 >>> (arr, res, size, constant);
	//Fuerzo una llamada Sincrona
	cudaThreadSynchronize();
	for (int index = 0; index < print_res; index++)
	{
		cout << res[index] << " , ";
	}
	cout << ']' << endl;
	
	cout << "Vector mas vector" << endl << '[';
	KernelVector_add_vector <<< ceil(size / 512.0), 512 >>> (arr, newArr, res, size);
	//Fuerzo una llamada Sincrona
	cudaThreadSynchronize();
	for (int index = 0; index < print_res; index++)
	{
		cout << res[index] << " , ";
	}
	cout << ']' << endl;

	cout << "Vector menos vector" << endl << '[';
	KernelVector_substract_vector << < ceil(size / 512.0), 512 >> > (arr, newArr, res, size);
	//Fuerzo una llamada Sincrona
	cudaThreadSynchronize();
	for (int index = 0; index < print_res; index++)
	{
		cout << res[index] << " , ";
	}
	cout << ']' << endl;


	cudaFree(arr);
	cudaFree(newArr);
	cudaFree(res);

	return 0;
}