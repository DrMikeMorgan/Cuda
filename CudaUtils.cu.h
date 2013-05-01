#ifndef CudaUtils_H
#define CudaUtils_H


namespace cudaMike
{
	/**
	*	Template class to hold data for scan (prefix sum) computation
	*/
	template<class T>
	struct prefixNode
	{
		T sum;
		T fromleft;
	};


	/**
	*	A logarithmic time parallel filter/stream compactor (assuming O(n) processors)
	*	@param	a		array to be filtered
	*	@param	n		length of a
	*	@param	output		array for results 
	*	@param	outLength	number of results 
	*	@param	proc		boolean functor, defines condition for filtering
	*/
	template<class T, class Process>
	__global__ void filter(T * a, int n, T * output, int * outLength, Process proc)
	{
		int i=threadIdx.x;
		extern __shared__ prefixNode<unsigned short> p[];	//woo, shared memory...
	
		if(i<n)
			p[i].sum = proc(a[i]) ? 1 : 0;
		__syncthreads();

		//compute prefix sum 
		int len = 1, lower = 0, higher = n, lowlen = n, highlen = __float2int_ru (n/2.f); 
		while(len<n) //upward loop
		{
			len*=2;
			if(i%2==0 && i<lowlen)
			{
				p[i/2+higher].sum= p[i+lower].sum; 
				if(i != lowlen-1) 
					p[i/2+higher].sum += p[i+lower+1].sum;
				p[i/2+higher].fromleft=0; // overkill but leave for now
			}
			lower = higher;
			higher += highlen;
			lowlen = highlen;
			highlen = __float2int_ru ( highlen/2.f );
			__syncthreads();	
		}

		while(len>1) //downward loop
		{
			len /= 2;
			highlen = lowlen;
			lowlen = __float2int_ru ( __int2float_rz(n)/len );		
			higher = lower;		
			lower -= lowlen;
			if(i%2==0 && i<lowlen) 
			{
				p[i+lower].fromleft = p[i/2+higher].fromleft;
				if(i!=lowlen -1)
					p[i+lower+1].fromleft= p[i/2+higher].fromleft + p[i+lower].sum;
			}
			__syncthreads(); //my lack of understanding compels me to write this everywhere
		}
		if(i<n)
			p[i].sum += p[i].fromleft;
		__syncthreads(); 

		//map to output array
		if(i<n && (i==0 && p[i].sum==1 || p[i].sum>p[i-1].sum) )
			output[p[i].sum-1]=a[i];
	 	*outLength = p[n-1].sum;
	}
}

#endif
