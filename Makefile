CUDAPATH = /opt/cuda
NVCC = nvcc
NVCCFLAGS = -O0 -g -I$(CUDAPATH)/include -L$(CUDA_ROOT_DIR)/lib64 -lpthread -std=c++11

all: checkers clean

checkers: main.cu kernel.h rules/rules.h rules/americanrules.h
	$(NVCC) $(NVCCFLAGS) main.cu -o checkers

clean:
	rm *.o
