import numpy as np
import pandas as pd

class Evaluation():
    def __init__(self, path, rate):
        """
        rate: test data percent
        """
        self._trajectorydata = pd.read_csv(path)
        ## TODO generate data
        return

    def getTrainData(self):
        """
        return:
           pandas dataframe with exactly same frame as what you read from csv
        """
        return

    def getTestData(self):
        return
    
if __name__ == "__main__":
    e = Evaluation("./NYC_Trajectory_Simplified.csv", 0.2)