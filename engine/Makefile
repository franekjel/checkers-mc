CUDAPATH = /opt/cuda
NVCC = nvcc
NVCCFLAGS = -O0 -g -lineinfo -I$(CUDAPATH)/include -L$(CUDA_ROOT_DIR)/lib64 -lpthread -lcudadevrt -std=c++11

all: checkers

checkers: main.cu kernel.h rules/rules.h rules/americanrules.h
	$(NVCC) $(NVCCFLAGS) main.cu -o checkers
