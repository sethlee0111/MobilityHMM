import pandas as pd
import numpy as np
from membership import MembershipVector

class Trajectory():
    def __init__(self, group, dataframe):
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
        self._data = data
        # initilize the membership vector
        self._member = MembershipVector(trajectorydata['UserID'].unique(), group)
        self._group = group


    def getData(self, groupId):
        """get training data based on group
        """
        rawdata = self._data
        length = np.asarray(rawdata.groupby('Trajectory').count()['Time'])
        ## Get probability of the user belongs to group
        proba = []
        userList = rawdata.groupby('Trajectory')['UserID'].unique().values
        for user in userList:
            proba.append(self._member.getUserProbByGroup(userId=user[0],groupId=groupId))
        data = rawdata.drop(columns='Trajectory')
        data = data.drop(columns='Venue category name')
        data = data.drop(columns='UserID')
        print(data)
        return (data.values, length, proba)

    def getTrajectoryByUser(self, userId):
        data = self._data.loc[self._data['UserID'] == userId]
        data = data.drop(columns='UserID')
        return data

    def getUserGroupProb(self, userId):
        return self._member.getProbByUserId(userId=userId)

    def getGroupMembers(self, groupId):
        return self._member.getUserByGroup(groupId=groupId)

if __name__ == "__main__":
    trajectorydata = pd.read_csv("./NYC_Trajectory_Simplified.csv")
    t = Trajectory(100,trajectorydata)
    data,length,proba = t.getData(1)
    #print(data)
    #print(length)
    #print(proba)
