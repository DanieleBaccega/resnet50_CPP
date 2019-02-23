INCLUDES = -I./include -I/opt/mxnet/include
LIBS = -L/opt/mxnet/lib -L/usr/local/cuda/lib64 -lmxnet
#LIBS = -L$(HOME)/opt/python_venv/lib/python2.7/site-packages/mxnet -lmxnet -lmklml_intel


resnet50: resnet50.cpp
	g++ -o resnet50 --std=c++11 resnet50.cpp $(INCLUDES) $(LIBS)

cleanall:
	rm -f resnet50
