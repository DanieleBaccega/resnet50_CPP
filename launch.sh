#!/bin/bash
FILE=./json_nets/resnet18_v2.json

make resnet50

./resnet50 $FILE
