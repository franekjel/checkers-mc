CUDAPATH = /opt/cuda
NVCC = nvcc
NVCCFLAGS = -O0 -g -I$(CUDAPATH)/include -L$(CUDA_ROOT_DIR)/lib64 -lpthread -std=c++11 -rdc=true

all: checkers clean

main:  kernel.cu kernel.h main.cu
        $(NVCC) -c $(NVCCFLAGS) main.cu

_rules: rules/rules.cu rules/rules.h
        $(NVCC) -c $(NVCCFLAGS) rules/rules.cu

americanrules: rules/americanrules.cu rules/americanrules.h
        $(NVCC) -c $(NVCCFLAGS) rules/americanrules.cu

kernel: kernel.cu kernel.h
        $(NVCC) -c $(NVCCFLAGS) kernel.cu

checkers: main kernel _rules americanrules
        $(NVCC) $(NVCCFLAGS) main.o kernel.o rules.o americanrules.o -o checkers

clean:
        rm *.o
