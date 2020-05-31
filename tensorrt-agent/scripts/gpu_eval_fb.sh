#!/usr/bin/env bash

# run framework trace to get acurate layer latency
# run system library trace to get acurate gpu kernel latency
# run system library trace with gpu metrics to get metrics of each cuda kernel
# https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference-7x
# achieved_occupancy: Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
# flop_count_sp: Number of single-precision floating-point operations executed by non-predicated threads (add, multiply, and multiply-accumulate). Each multiply-accumulate operation contributes 2 to the count. The count does not include special operations.

DATABASE_ADDRESS=$1
DATABASE_NAME=$2
BATCHSIZE=$3
MODELNAME=$4
OUTPUTFOLDER=$5
PRECISION=$6
NUMPREDS=1
DUPLICATE_INPUT=$(($NUMPREDS * $BATCHSIZE))
GPU_DEVICE_ID=0

cd ..

if [ -f tensorrt-agent ]; then
  rm tensorrt-agent
fi
go build

echo MODEL_TRACE
./tensorrt-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --database_address=$DATABASE_ADDRESS --publish --use_gpu --batch_size=$BATCHSIZE \
  --trace_level=MODEL_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID --precision=$PRECISION

export CUDA_LAUNCH_BLOCKING=1

echo FRAMEWORK_TRACE
./tensorrt-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu \
  --trace_level=FRAMEWORK_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID --precision=$PRECISION

echo SYSTEM_LIBRARY_TRACE
./tensorrt-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu \
  --trace_level=SYSTEM_LIBRARY_TRACE --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID --precision=$PRECISION

echo SYSTEM_LIBRARY_TRACE with GPU metric achieved_occupancy
./tensorrt-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=achieved_occupancy --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID --precision=$PRECISION

echo SYSTEM_LIBRARY_TRACE with GPU metric flop_count_sp
./tensorrt-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=flop_count_sp --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID --precision=$PRECISION

echo SYSTEM_LIBRARY_TRACE with GPU metrics dram_read_bytes,dram_write_bytes
./tensorrt-agent predict urls --model_name=$MODELNAME --duplicate_input=$DUPLICATE_INPUT --batch_size=$BATCHSIZE --database_address=$DATABASE_ADDRESS --publish --use_gpu \
  --trace_level=SYSTEM_LIBRARY_TRACE --gpu_metrics=dram_read_bytes,dram_write_bytes --database_name=$DATABASE_NAME --gpu_device_id=$GPU_DEVICE_ID --precision=$PRECISION
