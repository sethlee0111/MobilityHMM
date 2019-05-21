import pandas as pd
import numpy as np
from membership import MembershipVector

class Trajectory():
    def __init__(self, dataframe):
        self._df = dataframe

        data = self._df.drop(columns='Timegap')
        data = data[['UserID','Latitude','Longitude','Time','Venue category name','Trajectory','VenueID']]
        # convert Time to seconds
        data['Time'] = data['Time'].apply(lambda x: str(x)[-8:])
        data['Time'] = data['Time'].apply(lambda x: (int(str(x)[0:2]) * 60 + int(str(x)[3:5])) * 60 + int(str(x)[6:8]) )
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

    def getDataByUserGroup(self, users):
        """get training data based on group
        """
        data = self._df

        data = data.drop(columns='Timegap')
        data = data[['UserID','Latitude','Longitude','Time','Venue category name','Trajectory']]
        # convert Time to seconds
        data['Time'] = data['Time'].apply(lambda x: str(x)[-8:])
        data['Time'] = data['Time'].apply(lambda x: (int(str(x)[0:2]) * 60 + int(str(x)[3:5])) * 60 + int(str(x)[6:8]) )
        # encode venue category to id
        
        data = data.loc[data['UserID'].isin(users)]
        data = data.drop(columns='UserID')
        data['Venue category name'] = pd.Categorical(data['Venue category name'])
        data['VenueID'] = data['Venue category name'].cat.codes
        
        venuedDict = dict(enumerate(data['Venue category name'].cat.categories))
        data = data.drop(columns='Venue category name')
        length = np.asarray(data.groupby('Trajectory').count()['Time'])
        data = data.drop(columns='Trajectory')
        proba = np.ones(len(length))
        # print(data)

        return (data.values, length, proba, venuedDict)

    def getDataByUserGroupWithoutAssignVenueID(self, users):
        """get training data based on group without assign new VenueID
        """
        data = self._data
        
        data = data.loc[data['UserID'].isin(users)]
        data = data.drop(columns='UserID')
        length = np.asarray(data.groupby('Trajectory').count()['Time'])
        data = data.drop(columns='Trajectory')
        proba = np.ones(len(length))
        # print(data)

        return (data.values, length, proba) 

    def getDataByUserGroupAssignCustomVenueID(self, users, dic):
        """get training data based on group
        """
        data = self._df

        data = data.drop(columns='Timegap')
        data = data[['UserID','Latitude','Longitude','Time','Venue category name','Trajectory']]
        # convert Time to seconds
        data['Time'] = data['Time'].apply(lambda x: str(x)[-8:])
        data['Time'] = data['Time'].apply(lambda x: (int(str(x)[0:2]) * 60 + int(str(x)[3:5])) * 60 + int(str(x)[6:8]) )
        # encode venue category to id
        
        data = data.loc[data['UserID'].isin(users)]
        data = data.drop(columns='UserID')
        data['Venue category name'] 

        newDic = {}
        for key in dic:
            value = dic[key]
            newDic[value] = key
        #print(newDic)

        data['exist'] = data['Venue category name'].apply(lambda x : x in newDic)
        data = data.drop(data[data['exist']==False].index)
        data = data.drop(columns='exist')
        
        data['VenueID'] = data['Venue category name'].apply(lambda x : newDic[str(x)])
        
        data = data.drop(columns='Venue category name')
        
        length = np.asarray(data.groupby('Trajectory').count()['Time'])
        data = data.drop(columns='Trajectory')
        proba = np.ones(len(length))
        #print(length)
        #print(data)

        return (data.values, length, proba)

    def getDataWithAllGroups(self, member):
        """get test data based on group, with all the group proba
        """
        rawdata = self._data
        length = np.asarray(rawdata.groupby('Trajectory').count()['Time'])
        ## Get probability of the user belongs to group
        proba = []
        userList = rawdata.groupby('Trajectory')['UserID'].unique().values
        for user in userList:
            proba.append(member.getProbByUserId(userId=user[0]))
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
        data = data.drop(columns='Longitude')
        return (data.values, length)
    

    def getGaussianBaseModelData(self):
        """get base Gaussian model data -> only venue location coordinates
        """
        rawdata = self._data
        length = np.asarray(rawdata.groupby('Trajectory').count()['Time'])
        ## Get probability of the user belongs to group
        data = rawdata.drop(columns='Trajectory')
        data = data.drop(columns='UserID')
        data = data.drop(columns='VenueID')
        data = data.drop(columns='Time')
        return (data.values, length)


    def getTrajectoryByUser(self, userId):
        data = self._data.loc[self._data['UserID'] == userId]
        data = data.drop(columns='UserID')
        groups = data.groupby('Trajectory')
        arr = []
        for name, group in groups:
            #print(group.drop(columns='Trajectory'))
            arr.append(group.drop(columns='Trajectory').values)
        return arr
