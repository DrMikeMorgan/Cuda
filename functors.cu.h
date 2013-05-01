#ifndef functors_H
#define functors_H

namespace cudaMike
{

	/**
	*	functor for less than comparisons to a specified 'target' value	
	*/
	template <class T>
	struct less
	{
		less (const T& t):target(t){}
		T target; // the value to be compared to, initialised with constructor
		__device__ bool operator () (const T& t) {return t < target;}
	};

	/**
	*	functor for greater than comparisons to a specified 'target' value	
	*/
	template <class T>
	struct greater
	{
		greater (const T& t):target(t){}
		T target; // the value to be compared to, initialised with constructor
		__device__ bool operator () (const T& t) {return t > target;}
	};

	/**
	*	functor for less than or equal comparisons to a specified 'target' value	
	*/
	template <class T>
	struct less_eq
	{
		less_eq (const T& t):target(t){}
		T target; // the value to be compared to, initialised with constructor
		__device__ bool operator () (const T& t) {return t <= target;}
	};

	/**
	*	functor for greater than or equal comparisons to a specified 'target' value	
	*/
	template <class T>
	struct greater_eq
	{
		greater_eq (const T& t):target(t){}
		T target; // the value to be compared to, initialised with constructor
		__device__ bool operator () (const T& t) {return t >= target;}
	};

	/**
	*	functor for equality comparisons to a specified 'target' value	
	*/
	template <class T>
	struct equal
	{
		equal (const T& t):target(t){}
		T target; // the value to be compared to, initialised with constructor
		__device__ bool operator () (const T& t) {return t == target;}
	};

	/**
	*	functor for inequality comparisons to a specified 'target' value	
	*/
	template <class T>
	struct not_equal
	{
		not_equal (const T& t):target(t){}
		T target; // the value to be compared to, initialised with constructor
		__device__ bool operator () (const T& t) {return t != target;}
	};

	/**
	*	functor to check ranges - endpoints are exclusive by default, this can be altered by setting
	*	inclusive_lo and/or inclusive_hi to 1	
	*/
	template <class T, int inclusive_lo = 1, int inclusive_hi = 1>
	struct in_range
	{
		in_range (const T& lo, const T& hi):low(lo), high(hi){}
		T low, high; 	// the limits of the range, initialised with constructor
		__device__ bool operator () (const T& t) {return t > low && t< high;}			
	};

	/**
	*	Partial template specifications for in_range 
	*/
	template <class T>
	struct in_range<T,1,0>
	{
		in_range (const T& lo, const T& hi):low(lo), high(hi){}
		T low, high; 	// the limits of the range, initialised with constructor
		__device__ bool operator () (const T& t) {return t >= low && t< high;}			
	};

	template <class T>
	struct in_range<T,1,1>
	{
		in_range (const T& lo, const T& hi):low(lo), high(hi){}
		T low, high; 	// the limits of the range, initialised with constructor
		__device__ bool operator () (const T& t) {return t >= low && t<= high;}			
	};

	template <class T>
	struct in_range<T,0,1>
	{
		in_range (const T& lo, const T& hi):low(lo), high(hi){}
		T low, high; 	// the limits of the range, initialised with constructor
		__device__ bool operator () (const T& t) {return t > low && t<= high;}			
	};
}

#endif
