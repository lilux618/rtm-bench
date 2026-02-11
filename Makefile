# Makefile for 3D TTI RTM CUDA demo

APP      := rtm
SRC      := main.cu
NVCC     ?= nvcc
CUDA_ARCH ?= sm_70

CXXFLAGS := -O3 -std=c++14 -lineinfo
NVFLAGS  := -arch=$(CUDA_ARCH) $(CXXFLAGS)

.PHONY: all build run clean check-env help

all: build

build: $(APP)

$(APP): $(SRC) config.hpp model.hpp kernel.cuh
	$(NVCC) $(NVFLAGS) $< -o $@

run: $(APP)
	./$(APP)

clean:
	rm -f $(APP) image.bin image_slice.png

check-env:
	@command -v $(NVCC) >/dev/null 2>&1 && echo "Found $(NVCC): $$($(NVCC) --version | head -n 1)" || (echo "ERROR: $(NVCC) not found in PATH" && exit 1)

help:
	@echo "Targets:"
	@echo "  make / make build   Build the CUDA executable"
	@echo "  make run            Run the demo executable"
	@echo "  make clean          Remove build/result artifacts"
	@echo "  make check-env      Verify nvcc availability"
	@echo "Variables:"
	@echo "  CUDA_ARCH=sm_XX     GPU arch (default: sm_70)"
	@echo "  NVCC=<path>         nvcc compiler command"
