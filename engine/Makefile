CUDAPATH = /opt/cuda
NVCC = nvcc
NVCCFLAGS = -O3 -I$(CUDAPATH)/include -L$(CUDA_ROOT_DIR)/lib64 -lpthread -std=c++11 -Lgraphviz -lgvc -lcgraph -lcdt

all: checkers

checkers: main.cu kernel.h rules/rules.h rules/americanrules.h
	$(NVCC) $(NVCCFLAGS) main.cu -o checkers
