import json
import os
import statistics as stat
from enum import Enum
from collections import OrderedDict, defaultdict
from cpp_demangle import demangle


filename_model_trace = "trace_model_trace.json"
filename_framework_trace = "trace_framework_trace.json"
filename_sysmtem_library_trace = "trace_system_library_trace.json"

class traceLevel(Enum):
    MODEL_TRACE = 2
    FRAMEWORK_TRACE = 3
    SYSTEM_LIBRARY_TRACE = 5


class kernelTraceSummary:
    def __init__(self, framework, framework_version, model_name, batch_size, base_directory, precision="fp32", model_version="1.0", device="gpu", operating_system="ubuntu"):
        self.framework = framework
        self.framework_version = framework_version
        self.model_name = model_name
        self.model_version = model_version
        self.batch_size = batch_size
        self.precision = precision
        self.base_directory = base_directory
        self.device = device
        self.operating_system = operating_system

        self.keyword = 'predict' if framework == 'TensorRT' else 'c_predict'
        self.maxLenOperationName = -1
        self.filename = self.getKernelTraceFilename()
        self.trace_summary = self.setTraceSummary()

        self.modelSummary = None
        self.kernelSummary = None

    def getKernelTraceFilename(self):
        frameworkLowercase = self.framework.lower()
        resultDir = os.path.join(self.base_directory, frameworkLowercase, frameworkLowercase+"-agent", "results", self.framework, self.framework_version)
        filename = os.path.join(resultDir, self.model_name, self.model_version, self.batch_size, self.device, self.operating_system, self.precision, filename_sysmtem_library_trace)
        if not os.path.exists(filename):
            raise ValueError("SYSTEM_LIBRARY_TRACE file {} cannot be found".format(filename))

        return filename

    def setTraceSummary(self):
        with open(self.filename) as f:
            traces = json.load(f)

        traces = traces['data'][0]
        spans = traces['spans']

        batches = {}
        for span in spans:
            operationName = span['operationName']
            startTime = span['startTime']
            duration = span['duration']
            if operationName == self.keyword:
                batches[span['spanID']] = {'startTime': startTime, 'overallLatency': duration, 'kernels': []}
        batches = OrderedDict(sorted(batches.items(), key=lambda b: int(b[1]['startTime'])))

        for span in spans:
            operationName = span['operationName']
            startTime = span['startTime']
            parentID = self._getParentID(span['references'])
            duration = span['duration']

            if operationName != 'cuda_launch':
                continue

            for tag in span['tags']:
                if tag['key'] == 'correlation_id':
                    correlationID = tag['value']
                if tag['key'] == 'kernel':
                    try:
                        kernelName = demangle(tag['value'])
                    except:
                        kernelName = kernelName

            batches[parentID]['kernels'].append({
                'kernelName': kernelName,
                'correlationID': correlationID,
                'startTime': startTime,
                'duration': duration,
            })

        for _, batch in batches.items():
            batch['kernels'] = sorted(batch['kernels'], key=lambda x : int(x['correlationID']))

        return batches

    def getModelSummary(self):
        if self.modelSummary:
            return self.modelSummary

        self.modelSummary = []
        for _, info in self.trace_summary.items():
            self.modelSummary.append(info['overallLatency'])

        return self.modelSummary

    def getKernelSummary(self, sort_key="correlationID"):
        if self.kernelSummary:
            return self.kernelSummary

        self.kernelSummary = []
        for _, info in self.trace_summary.items():
            layers = info['kernels']
            if sort_key != "correlationID":
                layers = sorted(layers, key=lambda x : int(x[sort_key]))
            self.kernelSummary.append(layers)

        return self.kernelSummary

    @staticmethod
    def _getParentID(references):
        if not len(references):
            return -1

        return references[0]['spanID']


if __name__ == "__main__":
    # unitTime = defaultdict(int)
    # for res in result:
    #     bigLayer = res['operationName'][:5]
    #     # if bigLayer not in unitTime.keys():
    #     unitTime[bigLayer] += res['duration']

    # for k,v in unitTime.items():
    #     print(v)
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

    batch_sizes = [2**x for x in range(6)]
    precisions = [
        'fp32',
        # 'fp16',
        # 'int8',
    ]

    base_directory = '/home/jtang10/.gvm/pkgsets/go1.11/global/src/github.com/rai-project/'

    for model in models:
        tf_trace = kernelTraceSummary('TensorFlow', '1.12', model, str(1), base_directory)
        kernel_summary = tf_trace.getKernelSummary()

    kernel_summary = kernel_summary[0]
    summary = defaultdict(int)
    for k in kernel_summary:
        summary[k['kernelName']] += k['duration']

    summary = sorted(summary.items(), key=lambda kv : kv[1], reverse=True)
    for k, v in summary:
        print(k)
        print(v)
    # for model in models:
    #     for batch_size in batch_sizes:
    #         for precision in precisions:
    #             try:
    #                 tf_trace = kernelTraceSummary('TensorRT', '7.0.0', model, str(batch_size), base_directory, precision=precision)
    #                 tf_kernel_summary = tf_trace.getModelSummary()
    #                 # print(tf_kernel_summary)
    #                 print("{} with batch_size {} in {} average latency: {:.2f} ms.".format(model, batch_size, precision, stat.mean(tf_kernel_summary)/1000))
    #             except:
    #                 print("{} with batch size {} in {} file may not exist".format(model, batch_size, precision))