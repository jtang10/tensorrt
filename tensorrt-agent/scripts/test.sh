#!/usr/bin/env bash
declare -a models=(
  # AI_Matrix_Densenet121
  # BVLC-AlexNet
  # BVLC-GoogLeNet
  # Inception_v3
  Inception_v3_uff
  # Inception_v4
  # ResNet_v1_50
  # ResNet_v1_101
  # ResNet_v1_152
  # ResNet_v2_50
  # ResNet_v2_101
  # ResNet_v2_152
  # MobileNet_v1_0.5_128
  # MobileNet_v1_0.5_160
  # MobileNet_v1_0.5_192
  # MobileNet_v1_0.5_224
  # VGG16
  # VGG19

  # BVLC-Reference-CaffeNet
  # ResNet18
  # ResNet_v2_269
  # ResNeXt26-32x4d
  # ResNeXt50-32x4d
  # ResNeXt101-32x4d
  # SqueezeNet_v1
  # SqueezeNet_v1.1
  # WRN50_v2
  # Xception
)

declare -a traces=(
  # MODEL_TRACE
  FRAMEWORK_TRACE
)

declare -a precisions=(
  fp32
  fp16
  int8
)

BATCHSIZE=32
NUMPREDS=100
GPU_DEVICE_ID=0

cd ../..
make generate
cd tensorrt-agent

if [ -f tensorrt-agent ]; then
  rm tensorrt-agent
fi

echo "Building tensorrt-agent..."
go build
echo "building finished"

for trace in "${traces[@]}"; do
  for model in "${models[@]}"; do
    for precision in "${precisions[@]}"; do
      for ((b = 1; b <= $BATCHSIZE; b *= 2)); do
        echo "$model profiled at $trace with batch size $b at $precision"
        ./tensorrt-agent predict urls --model_name=$model --duplicate_input=$(($NUMPREDS * $b)) --use_gpu --batch_size=$b \
          --trace_level=$trace --gpu_device_id=$GPU_DEVICE_ID --precision $precision \
          # --database_name "trt_${precision}" --publish --database_address localhost
      done
    export CUDA_LAUNCH_BLOCKING=1
    done
  done
done

echo "Running kernel profiling for batch size 1 only"
for model in "${models[@]}"; do
  for precision in "${precisions[@]}"; do
      echo "$model profiled at SYSTEM_LIBRARY_TRACE with batch size 1 with $precision"
      ./tensorrt-agent predict urls --model_name=$model --duplicate_input=1 --use_gpu --batch_size=1 \
        --trace_level=SYSTEM_LIBRARY_TRACE --gpu_device_id=$GPU_DEVICE_ID --precision $precision \
        # --database_name "trt_${precision}" --publish --database_address localhost
  done
done

# echo SYSTEM_LIBRARY_TRACE with GPU metric achieved_occupancy
# for model in "${models[@]}"; do
#   for precision in "${precisions[@]}"; do
#       echo "$model profiled at SYSTEM_LIBRARY_TRACE with batch size $b"
#       ./tensorrt-agent predict urls --model_name=$model --duplicate_input=1 --use_gpu --batch_size=1 \
#         --trace_level=SYSTEM_LIBRARY_TRACE --gpu_device_id=$GPU_DEVICE_ID --publish --database_address localhost \
#         --database_name "trt_${precision}" --precision $precision --gpu_metrics=achieved_occupancy
#   done
# done

# echo SYSTEM_LIBRARY_TRACE with GPU metric flop_count_sp
# for model in "${models[@]}"; do
#   for precision in "${precisions[@]}"; do
#       echo "$model profiled at SYSTEM_LIBRARY_TRACE with batch size $b"
#       ./tensorrt-agent predict urls --model_name=$model --duplicate_input=1 --use_gpu --batch_size=1 \
#         --trace_level=SYSTEM_LIBRARY_TRACE --gpu_device_id=$GPU_DEVICE_ID --publish --database_address localhost \
#         --database_name "trt_${precision}" --precision $precision --gpu_metrics=flop_count_sp
#   done
# done

# echo SYSTEM_LIBRARY_TRACE with GPU metrics dram_read_bytes,dram_write_bytes
# for model in "${models[@]}"; do
#   for precision in "${precisions[@]}"; do
#       echo "$model profiled at SYSTEM_LIBRARY_TRACE with batch size $b"
#       ./tensorrt-agent predict urls --model_name=$model --duplicate_input=1 --use_gpu --batch_size=1 \
#         --trace_level=SYSTEM_LIBRARY_TRACE --gpu_device_id=$GPU_DEVICE_ID --publish --database_address localhost \
#         --database_name "trt_${precision}" --precision $precision --gpu_metrics=dram_read_bytes,dram_write_bytes
#   done
# done