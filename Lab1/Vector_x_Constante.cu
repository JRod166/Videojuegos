#include "stdafx.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

//DEVICE

__global__ void kernelVector_x_constant( float* arr, int n, int k )
{
	//Obtengo el indice del hilo fisico
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Mientras el hilo sea valido para la operación
	if( idx<n )
	{
		//Multiplico el elemento por la constante
		arr[ idx ] = arr[ idx ] * k;
	}
}

//HOST
void main()
{
	int size = 1000000;
	//Separo memoria en la RAM del HOST
	float* arr = new float[size];
	float* arr_DEVICE = NULL;

	//Inicializo el arreglo en el HOST
	for( int index = 0; index<size ; index++ )
	{
		arr[index] = index;
	}

	//Separo memoria en la RAM del DEVICE ( la misma cantidad de bytes que en el HOST )
	cudaMalloc((void**)&arr_DEVICE, size * sizeof(float));

	//Copio el bloque de memoria del HOST al DEVICE
	cudaMemcpy( arr_DEVICE, arr, size * sizeof(float), cudaMemcpyHostToDevice);
	
	///////////////////////// EJECUTO EL KERNEL DE CUDA ////////////////////////////
	//////// 512 Hilos
	//////// ceil(1000000/512) Bloques
	kernelVector_x_constant<<< ceil(size/512.0), 512 >>>( arr_DEVICE, size, 65 );
	//Fuerzo una llamada Sincrona
	cudaThreadSynchronize();

	//Copio mis datos ya procesados a la RAM del HOST 
	cudaMemcpy( arr, arr_DEVICE, size * sizeof(float), cudaMemcpyDeviceToHost);

	//Con una impresión de los primeros 100 visualizo el resultado
	for( int index = 0; index<100 ; index++ )
	{
		cout<<arr[index]<<endl;
	}
	
	//Libero memoria en la RAM del DEVICE
	cudaFree( arr_DEVICE );

	//Libero memoria en la RAM del HOST
	delete[] arr;
	cin.get();
}
