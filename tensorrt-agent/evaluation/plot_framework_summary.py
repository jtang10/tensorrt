import os
import matplotlib.pyplot as plt
import numpy as np
from evaluate_framework_summary import *


def plot_bar(plot_name):
    fig = plt.figure(figsize=(16, 9))
    # fig.suptitle("Inception-v3 TensorFlow vs. TensorRT", fontsize='xx-large', fontweight='bold')
    ax = fig.subplots(1, 1)
    labels = [str(bs) for bs in batch_sizes]
    x = np.arange(len(labels))

    width = 0.2
    rects1 = ax.bar(x - 1.65*width, tf_latencies_mean,     width, label='TF')
    rects2 = ax.bar(x - 0.55*width, trt_latencies_mean[0], width, label='TRT FP32')
    rects3 = ax.bar(x + 0.55*width, trt_latencies_mean[1], width, label='TRT FP16')
    rects4 = ax.bar(x + 1.65*width, trt_latencies_mean[2], width, label='TRT INT8')
    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Batch Size')
    # ax.set_title("Latency across percentile")
    # ax.set_xscale('log', basex=2); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 250)
    ax.legend()
    ax.grid(linestyle='dashed')

    def _autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    _autolabel(rects1)
    _autolabel(rects2)
    _autolabel(rects3)
    _autolabel(rects4)
    plt.savefig(plot_name)


models = [
    # "AI_Matrix_Densenet121",
    # "BVLC-AlexNet",
    # "BVLC-GoogLeNet",
    "Inception_v3",
    # "Inception_v3_Caffe",
    # "Inception_v4",
    # "ResNet_v1_50",
    # "ResNet_v1_101",
    # "ResNet_v1_152",
    # "ResNet_v2_50",
    # "ResNet_v2_101",
    # "ResNet_v2_152",
    # "MobileNet_v1_0.5_128",
    # "MobileNet_v1_0.5_160",
    # "MobileNet_v1_0.5_192",
    # "MobileNet_v1_0.5_224",
    # "VGG16",
    # "VGG19",
]

precisions = [
    'fp32',
    'fp16',
    'int8',
]

batch_sizes = [2**x for x in range(6)]
base_directory = '/home/jtang10/.gvm/pkgsets/go1.11/global/src/github.com/rai-project/'
num_runs = 100
current_dir = os.getcwd()
tf_np_array = "tf_test.npy"
trt_np_array = "trt_test.npy"
tf_np_array_path = os.path.join(current_dir, tf_np_array)
trt_np_array_path = os.path.join(current_dir, trt_np_array)

if os.path.exists(tf_np_array_path) and os.path.exists(trt_np_array_path):
    tf_latencies = np.load(tf_np_array_path)
    trt_latencies = np.load(trt_np_array_path)
else:
    tf_latencies = np.zeros((len(models), len(batch_sizes), num_runs))
    for i, model in enumerate(models):
        for j, batch_size in enumerate(batch_sizes):
            try:
                tf_trace = frameworkTraceSummary('TensorFlow', '1.12', model, str(batch_size), base_directory)
                tf_model_summary = tf_trace.getModelSummary()
                mean = stat.mean(tf_model_summary)/1000
                tf_latencies[i, j, :] = tf_model_summary
                print("{} with batch_size {} average latency: {:.2f} ms.".format(model, batch_size, mean))
            except:
                print("{} with batch size {} file may not exist".format(model, batch_size))

    trt_latencies = np.zeros((len(precisions), len(models), len(batch_sizes), num_runs))
    for i, precision in enumerate(precisions):
        for j, model in enumerate(models):
            for k, batch_size in enumerate(batch_sizes):
                try:
                    trt_trace = frameworkTraceSummary('TensorRT', '7.0.0', model, str(batch_size), base_directory, precision=precision)
                    trt_model_summary = trt_trace.getModelSummary()
                    mean = stat.mean(trt_model_summary)/1000
                    trt_latencies[i, j, k, :] = trt_model_summary
                    print("{} with batch_size {} in {} average latency: {:.2f} ms.".format(model, batch_size, precision, mean))
                except:
                    print("{} with batch size {} in {} file may not exist".format(model, batch_size, precision))

    np.save(tf_np_array, tf_latencies)
    np.save(trt_np_array, trt_latencies)

print(tf_latencies.shape)
print(trt_latencies.shape)
tf_latencies_mean = np.mean(tf_latencies, axis=-1).squeeze() / 1000
trt_latencies_mean = np.mean(trt_latencies, axis=-1).squeeze() / 1000
print(tf_latencies_mean.shape)
print(trt_latencies_mean.shape)

plot_bar('test_lantency')
