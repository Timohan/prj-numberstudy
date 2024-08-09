#/*!
#* \file Makefile
#* \brief Makefile for compiling
#*
#* Copyright of Timo hannukkala, Inc. All rights reserved.
#*
#* \author Timo Hannukkala <timohannukkala@hotmail.com>
#*/
TARGET:=prj-numberstudy-unittest
NVCC:=/usr/local/cuda/bin/nvcc
CXX:=g++
LDFLAGS+=-L/usr/local/cuda/targets/x86_64-linux/lib
LDFLAGS+=-lcudart -lcudadevrt
CXXFLAGS+=-g -Wall -pedantic -c -pipe -std=gnu++17 -W -D_REENTRANT -fPIC
CXXFLAGS+=-DDATABASEDIR=\"/home/timoh/pCloudDrive/economy\"
# set current make dir
CURRENT_DIR=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

src_SRCDIR:=$(CURRENT_DIR)src
src_SRCS:=$(wildcard $(src_SRCDIR)/*.cu)
src_OBJS:=$(src_SRCS:.cu=.o)

src_search_part_index_SRCDIR:=$(CURRENT_DIR)src/search_part_index
src_search_part_index_SRCS:=$(wildcard $(src_search_part_index_SRCDIR)/*.cu)
src_search_part_index_OBJS:=$(src_search_part_index_SRCS:.cu=.o)

src_original_search_part_index_SRCDIR:=$(CURRENT_DIR)../study/src/search_part_index
src_original_search_part_index_SRCS:=$(wildcard $(src_original_search_part_index_SRCDIR)/*.cu)
src_original_search_part_index_OBJS:=$(src_original_search_part_index_SRCS:.cu=.o)

default: $(src_OBJS) $(src_search_part_index_OBJS) $(src_original_search_part_index_OBJS)
	$(NVCC) --device-link $(src_OBJS) $(src_original_search_part_index_OBJS) $(src_search_part_index_OBJS) --output-file link.o
	$(CXX) $(src_OBJS) $(src_original_search_part_index_OBJS) $(src_search_part_index_OBJS) link.o $(LDFLAGS) -o $(TARGET)

$(src_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_SRCS) --output-directory $(src_SRCDIR)

$(src_search_part_index_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_search_part_index_SRCS) --output-directory $(src_search_part_index_SRCDIR)

$(src_original_search_part_index_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_original_search_part_index_SRCS) --output-directory $(src_original_search_part_index_SRCDIR)

clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f $(src_SRCDIR)/*.o
	rm -f $(src_search_part_index_SRCDIR)/*.o
	rm -f $(src_original_search_part_index_SRCDIR)/*.cpp
	rm -f $(src_original_search_part_index_SRCDIR)/*.o
