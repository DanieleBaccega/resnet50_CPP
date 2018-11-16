INCLUDES = -I./include -I/opt/incubator-mxnet/include
LIBS = -L/opt/incubator-mxnet/lib -lmxnet
#LIBS = -L$(HOME)/opt/python_venv/lib/python2.7/site-packages/mxnet -lmxnet -lmklml_intel


mlp:
	g++ -o mlp --std=c++11 mlp.cpp $(INCLUDES) $(LIBS)

gpu:
	g++ -o mlp_gpu --std=c++11 mlp_gpu.cpp $(INCLUDES) $(LIBS)

resnet: resnet.cpp
	g++ -o resnet --std=c++11 resnet.cpp $(INCLUDES) $(LIBS)

resnet50: resnet50.cpp
	g++ -o resnet50 --std=c++11 resnet50.cpp $(INCLUDES) $(LIBS)

mlp_gpu_resnet50: mlp_gpu_resnet50.cpp
	g++ -o mlp_gpu_resnet50 -g --std=c++11 mlp_gpu_resnet50.cpp $(INCLUDES) $(LIBS)

mlp_gpu_cp: mlp_gpu_cp.cpp
	g++ -o mlp_gpu_cp -g --std=c++11 mlp_gpu_cp.cpp $(INCLUDES) $(LIBS)

cleanall:
	rm -f mlp mlp_gpu resnet resnet50 mlp_gpu_resnet50 mlp_gpu_cp
