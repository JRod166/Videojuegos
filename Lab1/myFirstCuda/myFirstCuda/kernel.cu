#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void myFirstKernel()
{

}

void main()
{
	myFirstKernel << < 1, 1 >> > ();

	cin.get();
}

