#/*!
#* \file Makefile
#* \brief Makefile for compiling
#*
#* Copyright of Timo hannukkala, Inc. All rights reserved.
#*
#* \author Timo Hannukkala <timohannukkala@hotmail.com>
#*/
TARGET:=prj-numberstudy
CXX:=g++
CXXFLAGS+=-g -Wall -pedantic -c -pipe -std=gnu++17 -W -D_REENTRANT -fPIC
CXXFLAGS+=-I./src
CXXFLAGS+=-Werror
LDFLAGS_CPP:=$(PKGFLAGS)
# set current make dir
CURRENT_DIR=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all: default

src_SRCDIR:=$(CURRENT_DIR)src
src_SRCS:=$(wildcard $(src_SRCDIR)/*.cpp)
src_OBJS:=$(src_SRCS:.cpp=.o)

src_study_cuda_SRCDIR:=$(CURRENT_DIR)src/study_cuda
src_study_cuda_SRCS:=$(wildcard $(src_study_cuda_SRCDIR)/*.cpp)
src_study_cuda_OBJS:=$(src_study_cuda_SRCS:.cpp=.o)

src_study_SRCDIR:=$(CURRENT_DIR)src/study
src_study_SRCS:=$(wildcard $(src_study_SRCDIR)/*.cpp)
src_study_OBJS:=$(src_study_SRCS:.cpp=.o)

src_matrix_SRCDIR:=$(CURRENT_DIR)src/matrix
src_matrix_SRCS:=$(wildcard $(src_matrix_SRCDIR)/*.cpp)
src_matrix_OBJS:=$(src_matrix_SRCS:.cpp=.o)

data_SRCDIR:=$(CURRENT_DIR)src/data
data_SRCS:=$(wildcard $(data_SRCDIR)/*.cpp)
data_OBJS:=$(data_SRCS:.cpp=.o)

src_search_part_index_SRCDIR:=$(CURRENT_DIR)src/search_part_index
src_search_part_index_SRCS:=$(wildcard $(src_search_part_index_SRCDIR)/*.cpp)
src_search_part_index_OBJS:=$(src_search_part_index_SRCS:.cpo=.o)

src_loader_SRCDIR:=$(CURRENT_DIR)src/loader
src_loader_SRCS:=$(wildcard $(src_loader_SRCDIR)/*.cpp)
src_loader_OBJS:=$(src_loader_SRCS:.cpp=.o)

src_calculate_SRCDIR:=$(CURRENT_DIR)src/calculate
src_calculate_SRCS:=$(wildcard $(src_calculate_SRCDIR)/*.cpp)
src_calculate_OBJS:=$(src_calculate_SRCS:.cpo=.o)

src_options_SRCDIR:=$(CURRENT_DIR)src/options
src_options_SRCS:=$(wildcard $(src_options_SRCDIR)/*.cpp)
src_options_OBJS:=$(src_options_SRCS:.cpp=.o)


default: $(src_OBJS) $(src_options_OBJS) $(src_calculate_OBJS) $(src_loader_OBJS) $(data_OBJS) $(src_search_part_index_OBJS) $(src_study_cuda_OBJS) $(src_study_OBJS) $(src_matrix_OBJS)
	$(CXX) $(src_loader_OBJS) $(src_options_OBJS) $(src_calculate_OBJS) $(data_OBJS) $(src_OBJS) $(src_search_part_index_OBJS) $(src_study_cuda_OBJS) $(src_study_OBJS) $(src_matrix_OBJS) $(LDFLAGS_CPP) -o $(TARGET)

$(src_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_loader_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(data_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_search_part_index_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_study_cuda_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_study_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_matrix_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_calculate_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(src_options_OBJS):%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
	rm -f $(src_SRCDIR)/*.o
	rm -f $(src_SRCDIR)/*.cpp
	rm -f $(src_search_part_index_SRCDIR)/*.o
	rm -f $(src_search_part_index_SRCDIR)/*.cpp
	rm -f $(src_study_cuda_SRCDIR)/*.o
	rm -f $(src_study_cuda_SRCDIR)/*.cpp
	rm -f $(src_study_SRCDIR)/*.o
	rm -f $(src_study_SRCDIR)/*.cpp
	rm -f $(src_calculate_SRCDIR)/*.o
	rm -f $(src_calculate_SRCDIR)/*.cpp
	rm -f $(src_matrix_SRCDIR)/*.o
	rm -f $(src_matrix_SRCDIR)/*.cpp
	rm -f $(src_loader_SRCDIR)/*.o
	rm -f $(src_options_SRCDIR)/*.o
	rm -f $(CURRENT_DIR)*.o
