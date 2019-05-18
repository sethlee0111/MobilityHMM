import pandas as pd
import numpy as np
from membership import MembershipVector

class Trajectory():
    def __init__(self, dataframe):
        self._df = dataframe

        data = self._df.drop(columns='Timegap')
        data = data[['UserID','Latitude','Longitude','Time','Venue category name','Trajectory']]
        # convert Time to seconds
        data['Time'] = data['Time'].apply(lambda x: str(x)[-8:])
        data['Time'] = data['Time'].apply(lambda x: (int(str(x)[0:2]) * 60 + int(str(x)[3:5])) * 60 + int(str(x)[6:8]) )
        # encode venue category to id
        data['Venue category name'] = pd.Categorical(data['Venue category name'])
        data['VenueID'] = data['Venue category name'].cat.codes
        self._venuedDict = dict(enumerate(data['Venue category name'].cat.categories))
        # print(data['Time'].apply(lambda x: (int(str(x)[0:2]) * 24 + int(str(x)[3:5]))))
        data = data.drop(columns='Venue category name')
        self._data = data

    def getData(self, groupId, member):
        """get training data based on group
        """
        rawdata = self._data
        length = np.asarray(rawdata.groupby('Trajectory').count()['Time'])
        ## Get probability of the user belongs to group
        proba = []
        userList = rawdata.groupby('Trajectory')['UserID'].unique().values
        for user in userList:
            proba.append(member.getUserProbByGroup(userId=user[0],groupId=groupId))
        data = rawdata.drop(columns='Trajectory')
        data = data.drop(columns='UserID')
        return (data.values, length, proba)

    def getBaseModelData(self):
        """get base model data -> only venue id
        """
        rawdata = self._data
        length = np.asarray(rawdata.groupby('Trajectory').count()['Time'])
        ## Get probability of the user belongs to group
        data = rawdata.drop(columns='Trajectory')
        data = data.drop(columns='UserID')
        data = data.drop(columns='Latitude')
        data = data.drop(columns='Time')
        data = data.drop(columns='Longtitude')
        return (data.values, length)

    def getTrajectoryByUser(self, userId):
        data = self._data.loc[self._data['UserID'] == userId]
        data = data.drop(columns='UserID')
        groups = data.groupby('Trajectory')
        arr = []
        for name, group in groups:
            arr.append(group.values)
        return arr


if __name__ == "__main__":
    trajectorydata = pd.read_csv("./NYC_Trajectory_Simplified.csv")
    member = MembershipVector(trajectorydata['UserID'].unique(), 10)
    t = Trajectory(trajectorydata)
    #data,length,proba = t.getData(1, member)
    t.getTrajectoryByUser(1)
    #print(data)
    #print(length)
    #print(proba)
