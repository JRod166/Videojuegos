#pragma once
#ifdef __INTELLISENSE__
	void __syncthreads();
#endif

#include <stdio.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <chrono>


#define TILE 32

using namespace std;

//Para escribir documentos y comprobar la operación
void printMatrix(int* Matrix, int M, int N, string filename)
{
	ofstream writeFile;
	writeFile.open(filename);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			//cout << Matrix[(i * N) + j]<<'\t';
			writeFile << Matrix[(i * N) + j] << ';';
		}
		//cout << endl;
		writeFile << '\n';
	}
	writeFile.close();
	cout << filename << " escrito" << endl;
	return;
}

//DEVICE

__global__ void sumaColMatrizSharedKernel (int M, int N, int* MatrixD, int* ResD)
{
	__shared__ int blockCol[TILE];

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Escribiendo columna en memoria compartida
	blockCol[threadIdx.y] = MatrixD[(row * N) + col];

	__syncthreads();

	if (threadIdx.y == 0)
	{
		for (int i = 1; i < blockDim.y; ++i)
		{
			//Sumando columna del bloque
			atomicAdd(&blockCol[0], blockCol[i]);
		}

		//Sumando suma de columna de bloque con suma de columna general
		atomicAdd(&ResD[blockIdx.x], blockCol[0]);
	}

	return ;
}


__global__ void sumaColMatrizGlobalKernel (int M, int N, int* MatrixD, int* ResD) {

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row < M && col < N) {
		int tmp = 0;

		for (int i = 0; i < M; i++) {
			//Realizando sumas atómicas de cada registro de la columna
			atomicAdd(&ResD[col], MatrixD[(N * i) + col]);
		}
	}
}

//HOST
int main()
{
	cout << "Desea imprimir los resultados? (Y-N)" << endl;
	char printer = cin.get();
	bool printDocs = true;
	switch (printer)
	{
		case 'y':
		case 'Y':
			cout << "Escribiendo documentos" << endl;
			break;
		case 'n':
		case 'N':
			cout << "Continuando sin escribir documentos" << endl;
			printDocs = false;
			break;
		default:
			cout << "Opción incorrecta, escribiendo documentos" << endl;
			break;
	}
	cout << "Ingrese las dimensiones:" << endl << "M: ";
	cin.ignore();
	int M;
	cin >> M;
	cout << "N: ";
	int N;
	cin >> N;
	int block = 32;

	//Reservo espacio en memoria
	int size1 = M * N * sizeof(int);
	int size2 = N * sizeof(int);

	int* MatrixD;
	int* ResGlobalD;
	int* ResSharedD;

	cudaMallocManaged(&MatrixD, size1);
	cudaMallocManaged(&ResGlobalD, size2);
	cudaMallocManaged(&ResSharedD, size2);


	for (int i = 0; i < M; i++)
	{

		for (int j = 0; j < N; j++)
		{
			MatrixD[(i * N) + j] = (i * N) + j;
		}
	}
	cout << "Matriz instanciada ("<<M<<" x "<<N<<")" << endl;
	if (printDocs)
	{
		cout << "Escribiendo matriz, esto puede demorar" << endl;
		printMatrix(MatrixD, M, N, "matriz.csv");
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();	
	dim3 sharedDimGrid(N, M / block);
	dim3 sharedDimBlock(1, block);
	sumaColMatrizSharedKernel <<<sharedDimGrid, sharedDimBlock>>> (M, N, MatrixD, ResSharedD);
	cudaThreadSynchronize();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "Memoria compartida:" << endl;
	std::cout << "Tiempo de procesamiento = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
	if (printDocs)
	{
		cout << "Escribiendo respuesta en memoria compartida, esto puede demorar" << endl;
		printMatrix(ResSharedD, 1, N, "sumaShared.csv");
	}

	begin = std::chrono::steady_clock::now();
	dim3 globalDimBlock(block, block);
	dim3 globalDimGrid(ceil((float)N / (float)block), ceil((float)M / (float)block));
	sumaColMatrizGlobalKernel <<<globalDimGrid, globalDimBlock >>> (M, N, MatrixD, ResGlobalD);
	cudaThreadSynchronize();
	end = std::chrono::steady_clock::now();
	cout << "Memoria global:" << endl;
	std::cout << "Tiempo de procesamiento = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
	if (printDocs)
	{
		cout << "Escribiendo respuesta en memoria global, esto puede demorar" << endl;
		printMatrix(ResSharedD, 1, N, "sumaGlobal.csv");
	}
	return 0;
}