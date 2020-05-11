#pragma once
#ifdef __INTELLISENSE__
	void __syncthreads();
#endif

#include <stdio.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define TILE 32

using namespace std;

void printMatrix(int* Matrix, int M, int N)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cout << Matrix[(i * N) + j]<<'\t';
		}
		cout << endl;
	}
	return;
}

//DEVICE

__global__ void sumaColMatrizKernelShared (int M, int N, int* MatrixD, int* ResD)
{
	__shared__ int blockCol[TILE];

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	blockCol[threadIdx.y] = MatrixD[(row * N) + col];

	__syncthreads();

	if (threadIdx.y == 0)
	{
		for (int i = 1; i < blockDim.y; ++i)
		{
			blockCol[0] += blockCol[i];
		}

		ResD[blockIdx.x] += blockCol[0];
	}

	return ;
}


//HOST
int main()
{
	int M = 32;
	int N = 64;
	int block = 32;
	/*int* MatrixH = new int[M * N];
	int* ResH = new int[N];*/


	//Reservo espacio en memoria
	int size1 = M * N * sizeof(int);
	int size2 = N * sizeof(int);

	int* MatrixD;
	int* ResD;

	cudaMallocManaged(&MatrixD, size1);
	cudaMallocManaged(&ResD, size2);


	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			MatrixD[(i * N) + j] = (i * N) + j;
		}
	}

	dim3 dimGrid(N, M / block);
	dim3 dimBlock(1, block);
	sumaColMatrizKernelShared << <dimGrid, dimBlock >> > (M, N, MatrixD, ResD);
	

	printMatrix(MatrixD, M, N);
	printMatrix(ResD, N, 1);
	return 0;
}