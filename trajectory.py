import pandas as pd
import numpy as np

class Trajectory():
    ## generate the probablity of user group
    def __init__(self, group, dataframe):
        self._df = dataframe
        #self._member = MembershipVector(, group)
        self._group = group

    ## get training data based on group
    def getData(self, group):
        return (data, length, proba)

    def getTrajectoryByUser(self, userId):
        return

    def getUserGroupProb(self, userId):
        return

    def getGroupMembers(self, groupNumber):
        return
