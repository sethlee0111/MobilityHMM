import pandas as pd
import numpy as np

class MembershipVector():
    """
    userList: a list of user ID
    groupNum: a integer
    """
    def __init__(self, userList, groupNum):
        # randomly initilize the prob
        # init the userID and group Probability to a map of user
        self.dict = {}
        self.userList = userList
        for i in userList:
            self.dict[i] = np.random.dirichlet(np.ones(groupNum))
        print(self.dict)
        
    """ return group prob list of the input user
    """
    def getProbByUserId(self, userId):

        return self.dict[userId].tolist()

    """ return userID in the input group, group ID starts from 0
    """
    def getUserByGroup(self, groupId):
        groupList=[]
        for i in self.userList:
            userProb = self.dict[i].tolist()
            if userProb.index(max(userProb)) == groupId:
                groupList.append(i)
        return groupList
