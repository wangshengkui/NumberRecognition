import numpy as np


def calcDataSignature(filename):
    result = [];
    fs = open(filename);
    for i in range(32):
        line = fs.readline();
        for j in range(32):
            result.append(int(line[j]));
    return np.array(result);

def classify(currentFeature, featureSet, labels, K):
    numsOfDatas = featureSet.shape[0];
    curdataSet = np.repeat(currentFeature, numsOfDatas, axis = 0) - featureSet;
    distance = curdataSet ** 2;
    distanceSum = distance.sum(axis = 1); # calculate the eculid distance
    distanceSum = distanceSum ** (0.5);
    sortedIndex = distanceSum.argsort();
    result = [];
    weight = {};
    for i in range(K): # get the most similar K data
        weight[labels[sortedIndex[i]]] = weight.get(labels[sortedIndex[i]], 0) + 1;
    curMax = 0;
    ans = -1;
    for key in weight:
        if weight[key] > curMax:
            curMax = weight[key];
            ans = key;
    return ans;


print (calcDataSignature('data/digits/testDigits/0_1.txt').shape);