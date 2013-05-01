#include <stdio.h>
#include "CudaUtils.cu.h"
#include "functors.cu.h"

void testything()
{
	int a[]={13,17,16,19,12, 21, 14, 15, 20, 22};
	int n=10;
	int b[10];
	int * ad, * bd, * outlen;

	cudaMalloc((void**) &ad, n*sizeof(int));
	cudaMalloc((void**) &bd, n*sizeof(int));
	cudaMalloc((void**) &outlen, sizeof(int));
	
	cudaMemcpy(ad,a,n*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(bd,b,n*sizeof(int),cudaMemcpyHostToDevice);

	cudaMike::filter<<<1,n,256>>>(ad,n,bd,outlen, cudaMike::in_range<int>(16,21));

	int out;	
	cudaMemcpy(&out,outlen,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(b,bd,out*sizeof(int),cudaMemcpyDeviceToHost);
	
	for(int i=0; i<out; ++i)
		printf("%d:%d \t", i, b[i]);
	printf("\n");
	cudaFree(a);
	cudaFree(b);
	cudaFree(outlen);
}
