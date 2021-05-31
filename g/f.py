import numpy as np
import cv2
import random

nnOutput = [random.uniform(-100, 100) for i in range(9)]
nnOutput = [max(0, min(round(x), 1)) for x in nnOutput]
print(nnOutput)