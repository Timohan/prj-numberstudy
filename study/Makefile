#/*!
#* \file Makefile
#* \brief Makefile for compiling
#*
#* Copyright of Timo hannukkala, Inc. All rights reserved.
#*
#* \author Timo Hannukkala <timohannukkala@hotmail.com>
#*/
TARGET:=prj-numberstudy
NVCC:=/usr/local/cuda/bin/nvcc
CXX:=g++
LDFLAGS+=-L/usr/local/cuda/targets/x86_64-linux/lib
LDFLAGS+=-lcudart -lcudadevrt
CXXFLAGS+=-g -Wall -pedantic -c -pipe -std=gnu++17 -W -D_REENTRANT -fPIC
# set current make dir
CURRENT_DIR=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

src_SRCDIR:=$(CURRENT_DIR)src
src_SRCS:=$(wildcard $(src_SRCDIR)/*.cu)
src_OBJS:=$(src_SRCS:.cu=.o)

src_study_short_SRCDIR:=$(CURRENT_DIR)src/study_cuda
src_study_short_SRCS:=$(wildcard $(src_study_short_SRCDIR)/*.cu)
src_study_short_OBJS:=$(src_study_short_SRCS:.cu=.o)

src_study_SRCDIR:=$(CURRENT_DIR)src/study
src_study_SRCS:=$(wildcard $(src_study_SRCDIR)/*.cu)
src_study_OBJS:=$(src_study_SRCS:.cu=.o)

src_matrix_SRCDIR:=$(CURRENT_DIR)src/matrix
src_matrix_SRCS:=$(wildcard $(src_matrix_SRCDIR)/*.cu)
src_matrix_OBJS:=$(src_matrix_SRCS:.cu=.o)

src_data_SRCDIR:=$(CURRENT_DIR)src/data
src_data_SRCS:=$(wildcard $(src_data_SRCDIR)/*.cu)
src_data_OBJS:=$(src_data_SRCS:.cu=.o)

src_search_part_index_SRCDIR:=$(CURRENT_DIR)src/search_part_index
src_search_part_index_SRCS:=$(wildcard $(src_search_part_index_SRCDIR)/*.cu)
src_search_part_index_OBJS:=$(src_search_part_index_SRCS:.cu=.o)

src_loader_SRCDIR:=$(CURRENT_DIR)src/loader
src_loader_SRCS:=$(wildcard $(src_loader_SRCDIR)/*.cpp)
src_loader_OBJS:=$(src_loader_SRCS:.cpp=.o)

src_calculate_SRCDIR:=$(CURRENT_DIR)src/calculate
src_calculate_SRCS:=$(wildcard $(src_calculate_SRCDIR)/*.cu)
src_calculate_OBJS:=$(src_calculate_SRCS:.cu=.o)

src_options_SRCDIR:=$(CURRENT_DIR)src/options
src_options_SRCS:=$(wildcard $(src_options_SRCDIR)/*.cpp)
src_options_OBJS:=$(src_options_SRCS:.cpp=.o)

default: $(src_OBJS) $(src_options_OBJS) $(src_calculate_OBJS) $(src_search_part_index_OBJS) $(src_data_OBJS) $(src_study_OBJS) $(src_loader_OBJS) $(src_study_short_OBJS) $(src_matrix_OBJS)
	$(NVCC) --device-link $(src_calculate_OBJS) $(src_options_OBJS) $(src_search_part_index_OBJS) $(src_study_OBJS) $(src_OBJS) $(src_matrix_OBJS)  $(src_data_OBJS) $(src_study_short_OBJS) --output-file link.o
	$(CXX) $(src_loader_OBJS) $(src_options_OBJS) $(src_calculate_OBJS) $(src_search_part_index_OBJS) $(src_study_short_OBJS) $(src_study_OBJS) $(src_OBJS) $(src_matrix_OBJS) $(src_data_OBJS) link.o $(LDFLAGS) -o $(TARGET)

$(src_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_SRCS) --output-directory $(src_SRCDIR)

$(src_study_short_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_study_short_SRCS) --output-directory $(src_study_short_SRCDIR)

$(src_study_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_study_SRCS) --output-directory $(src_study_SRCDIR)

$(src_matrix_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_matrix_SRCS) --output-directory $(src_matrix_SRCDIR)

$(src_search_part_index_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_search_part_index_SRCS) --output-directory $(src_search_part_index_SRCDIR)

$(src_data_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_data_SRCS) --output-directory $(src_data_SRCDIR)

$(src_calculate_OBJS):%.o: %.cu
	$(NVCC) --device-c -DCUDA_COMPILE $(COMMONFLAGS) $(src_calculate_SRCS) --output-directory $(src_calculate_SRCDIR)

$(src_loader_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(COMMONFLAGS) -c $< -o $@

$(src_options_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(COMMONFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
	rm -f $(src_SRCDIR)/*.o
	rm -f $(src_SRCDIR)/*.cpp
	rm -f $(src_study_short_SRCDIR)/*.o
	rm -f $(src_study_short_SRCDIR)/*.cpp
	rm -f $(src_search_part_index_SRCDIR)/*.o
	rm -f $(src_search_part_index_SRCDIR)/*.cpp
	rm -f $(src_study_SRCDIR)/*.o
	rm -f $(src_study_SRCDIR)/*.cpp
	rm -f $(src_calculate_SRCDIR)/*.o
	rm -f $(src_calculate_SRCDIR)/*.cpp
	rm -f $(src_matrix_SRCDIR)/*.o
	rm -f $(src_matrix_SRCDIR)/*.cpp
	rm -f $(src_data_SRCDIR)/*.o
	rm -f $(src_data_SRCDIR)/*.cpp
	rm -f $(src_loader_SRCDIR)/*.o
	rm -f $(src_options_SRCDIR)/*.o
	rm -f $(CURRENT_DIR)*.o
