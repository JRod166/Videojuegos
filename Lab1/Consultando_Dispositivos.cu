#include "stdafx.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using namespace std;

void main()
{
	cudaDeviceProp prop;	

	int count = 0;

	cudaGetDeviceCount( &count );

	for( int index=0; index<count ;index++ )
	{
		cudaGetDeviceProperties(&prop, index);

		cout<<prop.name<<endl;
		cout<<prop.minor<<" - "<<prop.major<<endl;
		cout<<prop.clockRate<<endl;

	}	


	cin.get();
}
