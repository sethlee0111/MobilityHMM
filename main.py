import hmm as gmove
import numpy as np

X_example = np.array([[40, -70, 42320, 0],
					  [42, -71, 43989, 1],
					  [38, -72, 32324, 2],
					  [42, -70, 23222, 4],
					  [37, -71, 81999, 3]])
length_example = [3, 2]
weight_example = [1/2, 1/2]

model = gmove.GroupLevelHMM()
model._init(X_example, length_example, weight_example)
