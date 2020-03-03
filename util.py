import numpy as np

class Util():

    def __init__(self):
        return
    
    # helper function to make getting another batch of data easier
    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

    def filterDataset(self, datasetObject, indicies):
        newData = []
        newTargets = []

        for i in range(0, datasetObject.__len__()):
            if datasetObject.targets[i] in indicies:
                newData.append(datasetObject.data[i])
                newTargets.append(datasetObject.targets[i])

        datasetObject.data = newData
        datasetObject.targets = newTargets

        #datasetObject.data = np.vstack(datasetObject.data).reshape(-1, 3, 32, 32)
        #datasetObject.data = datasetObject.data.transpose((0, 2, 3, 1))  # convert to HWC

        datasetObject._load_meta()


