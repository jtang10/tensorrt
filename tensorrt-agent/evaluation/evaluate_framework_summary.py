import json
import os
import statistics as stat
from enum import Enum
from collections import OrderedDict, defaultdict


filename_model_trace = "trace_model_trace.json"
filename_framework_trace = "trace_framework_trace.json"
filename_sysmtem_library_trace = "trace_sysmtem_library_trace.json"

class traceLevel(Enum):
    MODEL_TRACE = 2
    FRAMEWORK_TRACE = 3
    SYSTEM_LIBRARY_TRACE = 5


class frameworkTraceSummary:
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
        self.filename = self.getFrameworkTraceFilename()
        self.trace_summary = self.setTraceSummary()

        self.modelSummary = None
        self.frameworkSummary = None

    def getFrameworkTraceFilename(self):
        frameworkLowercase = self.framework.lower()
        resultDir = os.path.join(self.base_directory, frameworkLowercase, frameworkLowercase+"-agent", "results", self.framework, self.framework_version)
        filename = os.path.join(resultDir, self.model_name, self.model_version, self.batch_size, self.device, self.operating_system, self.precision, filename_framework_trace)
        if not os.path.exists(filename):
            raise ValueError("FRAMEWORK_TRACE file {} cannot be found".format(filename))

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
                batches[span['spanID']] = {'startTime': startTime, 'overallLatency': duration, 'layers': []}
        batches = OrderedDict(sorted(batches.items(), key=lambda b: int(b[1]['startTime'])))

        for span in spans:
            operationName = span['operationName']
            startTime = span['startTime']
            parentID = self._getParentID(span['references'])
            duration = span['duration']

            # if operationName == 'c_predict':
            #     batches[parentID]['overallLatency'] = duration
            #     continue

            layerIdx = self._getLayerIdx(span['tags'])
            if layerIdx == -1:
                continue

            self.maxLenOperationName = max(self.maxLenOperationName, len(operationName))
            batches[parentID]['layers'].append({
                'layerIdx': layerIdx,
                'operationName': operationName,
                'startTime': startTime,
                'duration': duration,
            })

        for _, batch in batches.items():
            batch['layers'] = sorted(batch['layers'], key=lambda x : int(x['startTime']))

        return batches

    def getModelSummary(self):
        if self.modelSummary:
            return self.modelSummary

        self.modelSummary = []
        for _, info in self.trace_summary.items():
            self.modelSummary.append(info['overallLatency'])

        return self.modelSummary

    def getFrameworkSummary(self, sort_key="layerIdx"):
        if self.frameworkSummary:
            return self.frameworkSummary

        self.frameworkSummary = []
        for _, info in self.trace_summary.items():
            layers = info['layers']
            if sort_key != "layerIdx":
                layers = sorted(layers, key=lambda x : int(x[sort_key]))
            self.frameworkSummary.append(layers)

        return self.frameworkSummary

    def __str__(self):
        result = "="*10 + " Overall latency: " + "{:2f}ms " + "="*10 + "\n"
        for res in self.trace_summary:
            result += "Layer {0:<2}: {1:>{2}}, \tduration: {3:>3} us\n".format(res['layerIdx'], res['operationName'], self.maxLenOperationName, res['duration'])

        return result

    @staticmethod
    def _getLayerIdx(tags):
        for tag in tags:
            if tag['key'] == 'layer_sequence_index':
                return tag['value']

        return -1

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
        for batch_size in batch_sizes:
            try:
                tf_trace = frameworkTraceSummary('TensorFlow', '1.12', model, str(batch_size), base_directory)
                tf_model_summary = tf_trace.getModelSummary()
                # print(tf_model_summary)
                print("{} with batch_size {} average latency: {:.2f} ms.".format(model, batch_size, stat.mean(tf_model_summary)/1000))
            except:
                print("{} with batch size {} file may not exist".format(model, batch_size))

    for model in models:
        for batch_size in batch_sizes:
            for precision in precisions:
                try:
                    tf_trace = frameworkTraceSummary('TensorRT', '7.0.0', model, str(batch_size), base_directory, precision=precision)
                    tf_model_summary = tf_trace.getModelSummary()
                    # print(tf_model_summary)
                    print("{} with batch_size {} in {} average latency: {:.2f} ms.".format(model, batch_size, precision, stat.mean(tf_model_summary)/1000))
                except:
                    print("{} with batch size {} in {} file may not exist".format(model, batch_size, precision))
