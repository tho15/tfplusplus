####################################################################################
#
# simple make file to compile tensoflow application in C++
#
#####################################################################################
TF_ROOT = /home/tho/Development/Tools/tensorflow-1.10.0

TF_LIB_PATH=/usr/local/lib
TF_LIBS = -ltensorflow_cc -ltensorflow_framework
TF_INCLUDE = -I $(TF_ROOT) \
             -I $(TF_ROOT)/bazel-genfiles \
             -I $(TF_ROOT)/tensorflow/contrib/makefile/downloads \
             -I $(TF_ROOT)/tensorflow/contrib/makefile/downloads/eigen \
             -I $(TF_ROOT)/tensorflow/contrib/makefile/downloads/gemmlowp \
             -I $(TF_ROOT)/tensorflow/contrib/makefile/downloads/nsync/public \
             -I $(TF_ROOT)/tensorflow/contrib/makefile/downloads/protobuf/src

INTEL_LIB_PATH = /opt/intel/lib/intel64

INCLUDE_PATH = /usr/local/include $(TF_INCLUDE)

GCC_OPTS = -O3 -Wall -Wextra -std=c++14

fruit_cls: fruit_cls.o
	g++ -o fruit fruit_cls.o -L $(TF_LIB_PATH) -L $(INTEL_LIB_PATH) $(TF_LIBS)

fruit_cls.o: fruit_cls.cc
	g++ -c fruit_cls.cc -I $(INCLUDE_PATH) $(GCC_OPTS) 

clean:
	rm -f *.o fruit_cls
