#!/bin/bash
FILE=./json_nets/resnet50_v2_thumb.json

make resnet50

./resnet50 $FILE
