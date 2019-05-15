import pandas as pd
import numpy as np
from membership import MembershipVector

class Trajectory():
    def __init__(self, group, dataframe):
        self._df = dataframe
        # initilize the membership vector
        self._member = MembershipVector(trajectorydata['UserID'].unique(), group)
        self._group = group

    def generateTrajectoryData(self):
        data = self._df.drop(columns='Timegap')
        data = data[['UserID','Latitude','Longitude','Time','Venue category name','Trajectory']]
        data['Time'] = data['Time'].apply(lambda x: str(x)[-5:])
        return data

    def getData(self, groupId):
        """get training data based on group
        """
        rawdata = self.generateTrajectoryData()
        length = np.asarray(rawdata.groupby('Trajectory').count()['Time'])
        ## Get probability of the user belongs to group
        proba = []
        userList = rawdata.groupby('Trajectory')['UserID'].unique().values
        for user in userList:
            proba.append(self._member.getUserProbByGroup(userId=user[0],groupId=groupId))
        data = rawdata.drop(columns='Trajectory')
        data = data.drop(columns='UserID')
        return (data.values, length, proba)

    def getTrajectoryByUser(self, userId):
        return

    def getUserGroupProb(self, userId):
        return

    def getGroupMembers(self, groupId):
        return self._member.getUsersByGroup(groupId=groupId)

if __name__ == "__main__":
    trajectorydata = pd.read_csv("./NYC_Trajectory_Simplified.csv")
    t = Trajectory(100,trajectorydata)
    data,length,proba = t.getData(1)
    #print(data)
    #print(length)
    #print(proba)
