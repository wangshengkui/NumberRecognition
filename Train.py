import numpy as np


def calcDataSignature(filename):
    result = [];
    fs = open(filename);
    for i in range(32):
        line = fs.readline();
        for j in range(32):
            result.append(int(line[j]));
    return np.array(result);

print (calcDataSignature('data/digits/testDigits/0_1.txt').shape);