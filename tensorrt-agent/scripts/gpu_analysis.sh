#!/usr/bin/env bash

DATABASE_ADDRESS=$1
DATABASE_NAME=$2
BATCHSIZE=$3
MODELNAME=$4
OUTPUTFOLDER=$5

cd ..

if [ ! -d $OUTPUTFOLDER ]; then
  mkdir $OUTPUTFOLDER
fi

if [ -f tensorrt-agent ]; then
  rm tensorrt-agent
fi
go build

echo Start to run layer analysis

./tensorrt-agent evaluation layer info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --plot_all --output=$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_info

./tensorrt-agent evaluation layer aggre_info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --plot_all --output=$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/layer_aggre_info

echo Start to run gpu analysis

./tensorrt-agent evaluation gpu_kernel info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output=$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_info

./tensorrt-agent evaluation gpu_kernel name_aggre_info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output=$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_name_aggre_info

./tensorrt-agent evaluation gpu_kernel model_aggre_info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --output=$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_model_aggre_info

./tensorrt-agent evaluation gpu_kernel layer_aggre_info --database_address=$DATABASE_ADDRESS --database_name=$DATABASE_NAME --model_name=$MODELNAME --batch_size=$BATCHSIZE --sort_output --format=csv,table --plot_all --output=$OUTPUTFOLDER/$MODELNAME/$BATCHSIZE/gpu_kernel_layer_aggre_info
