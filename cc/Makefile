CC = gcc
CXX = g++

C_ARGS = -O3 -fopenmp -fPIC
CXX_ARGS = $(C_ARGS) -std=c++11

# INCLUDE_OPTIONS = -I. $(shell python -m pybind11 --includes)
INCLUDE_OPTIONS = -I.
LINK_OPTIONS = 

# PYBIND_SUFFIX_OPTIONS = $(shell python3-config --extension-suffix)

# all: oracle$(PYBIND_SUFFIX_OPTIONS) libloporacle.so
all: libloporacle.so

# callable function for Julia
libloporacle.so: oracle/oracleexport.cpp
	$(CXX) $(CXX_ARGS) -fPIC -shared $(INCLUDE_OPTIONS) -o $@ $^ $(LINK_OPTIONS)

clean:
	@rm -f oracle/*.o
	@rm -f *.so
