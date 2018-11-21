/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Xin Li yakumolx@gmail.com
 */
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;

Symbol ConvolutionNoBias(const std::string& symbol_name,
                         Symbol data,
                         Symbol weight,
                         Shape kernel,
                         int num_filter,
                         Shape stride = Shape(),
                         Shape dilate = Shape(),
                         Shape pad = Shape(),
                         int num_group = 1,
                         int64_t workspace = 256) {
  return Operator("Convolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("dilate", dilate)
      .SetParam("pad", pad)
      .SetParam("num_group", num_group)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", true)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .CreateSymbol(symbol_name);
}

Symbol residual_unit(Symbol data,
		     int num_filter,
		     Shape stride,
		     bool dim_match,
		     const std::string &name,
		     bool bottle_neck=true,
		     mx_float bn_mom=0.9,
	     	     int workspace=256) {

  Symbol gamma(name + "_gamma"), beta(name + "_beta");
  Symbol mmean(name + "_mmean"), mvar(name + "_mvar");
  Symbol shortcut_w(name + "_proj_w");
  Symbol shortcut;

  if(bottle_neck){
    Symbol bn1 = BatchNorm(name + "_bn1", data, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
    Symbol act1 = Activation(name + "_relu1", bn1, "relu");
    Symbol conv1 = ConvolutionNoBias(name + "conv_1", act1, shortcut_w, Shape(1, 1), int(num_filter*0.25), Shape(1, 1), Shape(), Shape(0, 0), 1, workspace);

    Symbol bn2 = BatchNorm(name + "_bn2", conv1, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
    Symbol act2 = Activation(name + "_relu2", bn2, "relu");
    Symbol conv2 = ConvolutionNoBias(name + "conv_2", act2, shortcut_w, Shape(3, 3), int(num_filter*0.25), stride, Shape(), Shape(1, 1), 1, workspace);

    Symbol bn3 = BatchNorm(name + "_bn3", conv2, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
    Symbol act3 = Activation(name + "_relu3", bn3, "relu");
    Symbol conv3 = ConvolutionNoBias(name + "conv_3", act3, shortcut_w, Shape(1, 1), num_filter, Shape(1, 1), Shape(), Shape(0, 0), 1, workspace);

    if(dim_match){
      shortcut = data;
    }
    else{
      shortcut = ConvolutionNoBias(name + "_sc", act1, shortcut_w, Shape(1, 1), num_filter, stride, Shape(), Shape(), 1, workspace);
    }

    return conv3 + shortcut;
  }
  else{
    Symbol bn1 = BatchNorm(name + "_bn1", data, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
    Symbol act1 = Activation(name + "_relu1", bn1, "relu");
    Symbol conv1 = ConvolutionNoBias(name + "conv_1", act1, shortcut_w, Shape(3, 3), num_filter, stride, Shape(), Shape(1, 1), 1, workspace);

    Symbol bn2 = BatchNorm(name + "_bn2", conv1, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
    Symbol act2 = Activation(name + "_relu2", bn2, "relu");
    Symbol conv2 = ConvolutionNoBias(name + "conv_2", act2, shortcut_w, Shape(3, 3), num_filter, Shape(1, 1), Shape(), Shape(1, 1), 1, workspace);

    if(dim_match){
      shortcut = data;
    }
    else{
      shortcut = ConvolutionNoBias(name + "_sc", act1, shortcut_w, Shape(1, 1), num_filter, stride, Shape(), Shape(), 1, workspace);
    }

    return conv2 + shortcut;
  }
}

Symbol resnet(vector<int> units,
	      int num_stages,
      	      vector<int> filter_list,
	      int num_classes,
	      vector<int> image_shape,
	      bool bottle_neck=true,
	      mx_float bn_mom=0.9,
	      int workspace=256,
	      string dtype="float32"){

  int num_unit = units.size();
  assert(num_unit == num_stages);

  Symbol gamma("gamma"), beta("beta");
  Symbol mmean("mmean"), mvar("mvar");
  Symbol shortcut_w("projbody_w"), weight("weight");
  Symbol bias("bias");
  Symbol data("data");
  Symbol label("label");
  Symbol body;
  Symbol bn1, relu1, pool1, flat, fc1;
  int channel, height, width;

  channel = image_shape[0];
  height = image_shape[1];
  width = image_shape[2];

  data = BatchNorm("bn_data", data, gamma, beta, mmean, mvar, 2e-5, bn_mom, true);
  if(height <= 32){
    body = ConvolutionNoBias("conv0", data, shortcut_w, Shape(3, 3), filter_list[0], Shape(1, 1), Shape(), Shape(1, 1), 1, workspace);
  }
  else{
    body = ConvolutionNoBias("conv0", data, shortcut_w, Shape(7, 7), filter_list[0], Shape(2, 2), Shape(), Shape(3, 3), 1, workspace);
    body = BatchNorm("bn0", body, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
    body = Activation("relu0", body, "relu");
    body = Pooling(body, Shape(3, 3), PoolingPoolType::kMax, false, false, PoolingPoolingConvention::kValid, Shape(2, 2), Shape(1, 1));
  }

  for(int i=0;i<num_stages;i++){
    Shape stride;
    if(i == 0){
      stride = Shape(1, 1);
    }
    else{
      stride = Shape(2, 2);
    }
    body = residual_unit(body, filter_list[i+1], stride, false, "stage" + to_string(i+1) + "_unit1", bottle_neck, 0.9, workspace);
    for(int j=0;j<units[i]-1;j++){
      body = residual_unit(body, filter_list[i+1], Shape(1, 1), true, "stage" + to_string(i+1) + "_unit1" + to_string(j+2), bottle_neck, 0.9, workspace);
    }
  }

  bn1 = BatchNorm("bn1", body, gamma, beta, mmean, mvar, 2e-5, bn_mom, false);
  relu1 = Activation("relu1", bn1, "relu");
  pool1 = Pooling("pool1", relu1, Shape(7, 7), PoolingPoolType::kAvg, true);
  flat = Flatten(pool1);
  fc1 = FullyConnected("fc1", flat, weight, bias, num_classes);

  return SoftmaxOutput("softmax", fc1, label);
}

Symbol get_symbol(int num_classes,
                  int num_layers,
		  vector<int> image_shape,
		  int conv_workspace=256,
		  string dtype="float32"){

  int channel, height, width;
  int num_stages;
  int per_unit;
  vector<int> units, filter_list;
  bool bottle_neck;

  channel = image_shape[0];
  height = image_shape[1];
  width = image_shape[2];

  if(height <= 28){
    num_stages = 3;
    if(((num_layers-2) % 9 == 0) && (num_layers >= 164)){
      per_unit = (num_layers-2) / 9;
      filter_list = {16, 64, 128, 256};
      bottle_neck = true;
    }
    else if(((num_layers-2) % 6 == 0) && (num_layers < 164)){
      per_unit = (num_layers-2) / 6;
      filter_list = {16, 16, 32, 64};
      bottle_neck = false;
    }
    else{
      throw invalid_argument("No experiments done on num_layers %d, you can do it yourself.");
    }
    units = {per_unit, per_unit, per_unit};
  }
  else{
    if(num_layers >= 50){
      filter_list = {64, 256, 512, 1024, 2048};
      bottle_neck = true;
    }
    else{
      filter_list = {64, 64, 128, 256, 512};
      bottle_neck = false;
    }
    num_stages = 4;
    if(num_layers == 18){
      units = {2, 2, 2, 2};
    }
    else if(num_layers == 34){
      units = {3, 4, 6, 3};
    }
    else if(num_layers == 50){
      units = {3, 4, 6, 3};
    }
    else if(num_layers == 101){
      units = {3, 4, 23, 3};
    }
    else if(num_layers == 152){
      units = {3, 8, 36, 3};
    }
    else if(num_layers == 200){
      units = {3, 24, 36, 3};
    }
    else if(num_layers == 269){
      units = {3, 30, 48, 8};
    }
    else{
      throw invalid_argument("No experiments done on num_layers %d, you can do it yourself.");
    }
  }

  return resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck, 0.9, conv_workspace, dtype);
}




int main(int argc, char** argv) {
  const int image_size = 28;
  const int channels = 1;
  const int batch_size = 128;
  const int max_epoch = 400;
  const float learning_rate = 0.001;
  const float weight_decay = 0.0005;


  /*auto train_iter = MXDataIter("ImageRecordIter")
	.SetParam("path_imglist", "./cifar10/cifar10_train.lst")
        .SetParam("path_imgrec", "./cifar10/cifar10_train.rec")
        .SetParam("rand_crop", 1)
        .SetParam("rand_mirror", 1)
        .SetParam("data_shape", Shape(3, 32, 32))
        .SetParam("batch_size", batch_size)
        .SetParam("shuffle", 1)
        .SetParam("preprocess_threads", 1)
	.SetParam("pad", 2)
        .CreateDataIter();

  auto val_iter = MXDataIter("ImageRecordIter")
	.SetParam("path_imglist", "./cifar10/cifar10_val.lst")
	.SetParam("path_imgrec", "./cifar10/cifar10_val.rec")
        .SetParam("rand_crop", 0)
        .SetParam("rand_mirror", 0)
        .SetParam("data_shape", Shape(3, 32, 32))
        .SetParam("batch_size", batch_size)
        .SetParam("round_batch", 0)
	.SetParam("preprocess_threads", 1)
	.SetParam("pad", 2)
        .CreateDataIter();*/

  auto train_iter = MXDataIter("MNISTIter")
      .SetParam("image", "../data/mnist_data/train-images-idx3-ubyte")
      .SetParam("label", "../data/mnist_data/train-labels-idx1-ubyte")
      .SetParam("batch_size", batch_size)
      //.SetParam("flat", 1)
      .CreateDataIter();

  auto val_iter = MXDataIter("MNISTIter")
      .SetParam("image", "../data/mnist_data/t10k-images-idx3-ubyte")
      .SetParam("label", "../data/mnist_data/t10k-labels-idx1-ubyte")
      .SetParam("batch_size", batch_size)
      //.SetParam("flat", 1)
      .CreateDataIter();

  //auto net = get_symbol(10, 50, {channels, image_size, image_size});
  auto net = Symbol::Load("resnet50v2.json");

  Context ctx = Context::gpu();  // Use GPU for training

  std::map<string, NDArray> args;
  args["data"] = NDArray(Shape(batch_size, channels, image_size, image_size), ctx);
  args["label"] = NDArray(Shape(batch_size), ctx);
  //Let MXNet infer shapes other parameters such as weights
  net.InferArgsMap(ctx, &args, args);

  //Initialize all parameters with uniform distribution U(-0.01, 0.01)
  auto initializer = Uniform(0.01);
  for (auto& arg : args) {
    //arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }

  //Create sgd optimizer
  Optimizer* opt = OptimizerRegistry::Find("adam");
  opt->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);

  //Create executor by binding parameters to the model
  auto *exec = net.SimpleBind(ctx, args);
  auto arg_names = net.ListArguments();
  Accuracy train_acc;

  //Start training
  for (int iter = 0; iter < max_epoch; ++iter) {
    int samples = 0;
    train_iter.Reset();
    train_acc.Reset();

    auto tic = chrono::system_clock::now();
    while (train_iter.Next()) {
      samples += batch_size;
      auto data_batch = train_iter.GetDataBatch();
      //Set data and label
      data_batch.data.CopyTo(&args["data"]);
      data_batch.label.CopyTo(&args["label"]);
      NDArray::WaitAll();

      //Compute gradients
      exec->Forward(true);
      exec->Backward();
      train_acc.Update(data_batch.label, exec->outputs[0]);
      //Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }
    }
    auto toc = chrono::system_clock::now();


    float duration = chrono::duration_cast<chrono::milliseconds>(toc - tic).count() / 1000.0;
    LG << "Epoch: " << iter << " " << samples/duration << " samples/sec Training accuracy: " << train_acc.Get();
  }

  Accuracy acc;
  val_iter.Reset();
  while (val_iter.Next()) {
	auto data_batch = val_iter.GetDataBatch();
	data_batch.data.CopyTo(&args["data"]);
	data_batch.label.CopyTo(&args["label"]);
	// Forward pass is enough as no gradient is needed when evaluating
	exec->Forward(false);
	acc.Update(data_batch.label, exec->outputs[0]);
  }
  LG << "Accuracy: " << acc.Get();

  delete exec;
  MXNotifyShutdown();
  return 0;
}
