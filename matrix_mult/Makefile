CXX = g++
CC = g++
NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc

CFLAGS=-O3 -I. -I/usr/local/cuda/include 
CXXFLAGS=-O3 -I. -I/usr/local/cuda/include

LIBS = -lm -lcuda -lcudart

SRC =  main.o matrix_mul.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) -allow-unsupported-compiler $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

matrix_mul: $(SRC) 
	$(CC) -o matrix_mult  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o matrix_mult

