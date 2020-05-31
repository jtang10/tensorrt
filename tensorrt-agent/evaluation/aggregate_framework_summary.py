import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, OrderedDict
from evaluate_framework_summary import frameworkTraceSummary


model = "Inception_v3"


precisions = [
    'fp32',
    'fp16',
    'int8',
]

batch_sizes = [2**x for x in range(6)]
base_directory = '/home/jtang10/.gvm/pkgsets/go1.11/global/src/github.com/rai-project/'
num_runs = 100
current_dir = os.getcwd()

def aggregate_framework_summary(trace):
    layers = trace['layers']
    batch_layer = OrderedDict()
    for layer in layers:
        operationName = layer['operationName']
        startTime = layer['startTime']
        duration = layer['duration']
        unitName = operationName.split('/')
        if len(unitName) == 1:
            key = 'Reformat'
            if key not in batch_layer.keys():
                batch_layer[key] = []
            batch_layer[key].append({'auxName': operationName, 'duration': duration, 'startTime': startTime})
            continue

        if unitName[1] != 'InceptionV3':
            key = 'Logits & Prediction'
            if key not in batch_layer.keys():
                batch_layer[key] = []
            batch_layer[key].append({'auxName': operationName, 'duration': duration, 'startTime': startTime})
            continue

        startIdx = 2
        auxName = '/'.join(unitName[(startIdx+1):])
        unitName = unitName[startIdx]
        if unitName not in batch_layer.keys():
            batch_layer[unitName] = []
        batch_layer[unitName].append({'auxName': auxName, 'duration': duration, 'startTime': startTime})

    actual_duration = OrderedDict()
    for name, members in batch_layer.items():
        if len(members) < 2:
            actual_duration[name] = members[0]['duration']
        else:
            duration = members[-1]['startTime'] + members[-1]['duration'] - members[0]['startTime']
            actual_duration[name] = duration

    return batch_layer, actual_duration

tf_summaries = []
tf_trace = frameworkTraceSummary('TensorFlow', '1.12', model, str(1), base_directory)
tf_trace_summary = tf_trace.trace_summary
for batch_evaluate in tf_trace_summary.values():
    summary, actual_duration = aggregate_framework_summary(batch_evaluate)
    # for k, v in summary.items():
    #     print(k, v)
    # for k, v in actual_duration.items():
    #     print(k, v)
    # print('total latency:', sum(list(actual_duration.values())))
    # break
    tf_summaries.append(list(actual_duration.values()))

for k, v in summary.items():
    print(k, v)
labels = summary.keys()
tf_summaries = np.array(tf_summaries)
# print(tf_summaries.shape)

trt_summaries = []
for precision in precisions:
    trt_sum = []
    trt_trace = frameworkTraceSummary('TensorRT', '7.0.0', model, str(1), base_directory, precision=precision)
    trt_trace_summary = trt_trace.trace_summary
    for batch_evaluate in trt_trace_summary.values():
        summary, actual_duration = aggregate_framework_summary(batch_evaluate)
        # for k, v in summary.items():
        #     print(k, v)
        # for k, v in actual_duration.items():
        #     print(k, v)
        # print('total latency:', sum(list(actual_duration.values())))
        # break
        trt_sum.append(list(actual_duration.values()))
    trt_summaries.append(trt_sum)
trt_summaries = np.array(trt_summaries)
# print(trt_summaries.shape)



data = np.concatenate((np.expand_dims(tf_summaries, 0), trt_summaries))
data = np.mean(data, axis=1)
print(data)
# data = data / 1000.0
data_cum = data.cumsum(axis=1)

fig, ax = plt.subplots(figsize=(16, 9))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.set_xlim(0, np.sum(data, axis=1).max())

category_colors = plt.get_cmap('tab20c')(np.linspace(0, 1, data.shape[1]))
labels2 = ['TF', 'TRT FP32', 'TRT FP16', 'TRT INT8',]
for i, (colname, color) in enumerate(zip(labels, category_colors)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    ax.barh(labels2, widths, left=starts, height=0.5, label=colname, color=color, edgecolor='darkgrey', fontsize='medium')
    xcenters = starts + widths / 2

    r, g, b, _ = color
    text_color = 'white' if r * g * b < 0.07 else 'black'
    for y, (x, c) in enumerate(zip(xcenters, widths)):
        ax.text(x, y, str(int(c)), ha='center', va='center',
                color=text_color, fontsize='x-small')
ax.legend(ncol=len(labels)//2+1, bbox_to_anchor=(0, 1),
            loc='lower left', fontsize='large')
plt.show()