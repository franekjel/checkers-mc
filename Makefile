CUDAPATH = /opt/cuda
NVCC = nvcc
NVCCFLAGS = -O2 -I$(CUDAPATH)/include -L$(CUDA_ROOT_DIR)/lib64 -lpthread -std=c++11

all: checkers clean

main:  kernel.cu kernel.h main.cu
	$(NVCC) -c $(NVCCFLAGS) main.cu

kernel: kernel.cu kernel.h
	$(NVCC) -c $(NVCCFLAGS) kernel.cu

checkers: main kernel kernel.h
	$(NVCC) $(NVCCFLAGS) main.o kernel.o -o checkers

clean:
	rm *.o
