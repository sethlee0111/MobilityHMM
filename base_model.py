import pandas as pd
import numpy as np
from hmmlearn import hmm
from trajectory import Trajectory

trajectorydata = pd.read_csv("./NYC_Trajectory_Simplified.csv")
t = Trajectory(trajectorydata)
data, length = t.getBaseModelData()

remodel = hmm.GaussianHMM(n_components=10, covariance_type="full", n_iter=100)
remodel.fit(data, length)

