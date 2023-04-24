EXENAME = AMM

CUSRCS  = $(wildcard *.cu)
OBJS    = $(CUSRCS:.cu=.o)

CUDA_PATH  = /usr/local/cuda-12.0
NVCC       = $(CUDA_PATH)/bin/nvcc
NVFLAGS    = -O3 -std=c++17 --generate-code arch=compute_80,code=sm_80 -Xcompiler -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc --use_fast_math
LDFLAGS    = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcurand

build : $(EXENAME)

$(EXENAME): $(OBJS)
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $(EXENAME) $(OBJS)

%.o : %.cu
	$(NVCC) $(NVFLAGS)  -c $^

clean:
	$(RM) *.o $(EXENAME)
