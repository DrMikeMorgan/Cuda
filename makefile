NVCC		:= /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda/lib64

all: main.cu main.cpp
	$(NVCC) -o exe main.cu main.cpp

clean:
	rm -rf exe 
