import numpy as np
from os.path import dirname, join, basename
from glob import glob

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
    curdataSet = np.tile(currentFeature, (numsOfDatas, 1)) - featureSet;
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

def predict(filename, featureSet, labels, K):
    sign = calcDataSignature(filename);
    return classify(sign, featureSet, labels, K);

def initFeatureSet():
    cnt = 0;
    labels = [];
    featureSet = [];
    for i in glob(join(dirname(__file__) + '/data/digits/trainingDigits', '*.txt')):
        fileName = i.split('\\')[1].split('.')[0];
        labels.append(int(fileName.split('_')[0]));
        featureSet.append(calcDataSignature(i));

    return np.array(featureSet), labels;


featureSet, labels = initFeatureSet();
mK = 3;
errors = 0;
total = 0;
for i in glob(join(dirname(__file__) + '/data/digits/testDigits', '*.txt')):
    predictedRes = predict(i, featureSet, labels, mK);
    fileName = i.split('\\')[1].split('.')[0];
    actual = int(fileName.split('_')[0]);
    print("predict value %d, actual value %d" % (predictedRes, actual));
    total = total + 1;
    if predictedRes != actual:
        errors = errors + 1;

print("total %d    errors %d" % (total, errors));

