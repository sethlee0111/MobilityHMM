import pandas as pd
import numpy as np

class MembershipVector():
    def __init__(self, userList, groupNum):
        """
        randomly initilize the prob
        init the userID and group Probability to a map of user'

        Args:
            userList: a list of user ID
            groupNum: a integer
        """
        self.dict = {}
        self.userList = userList
        for i in userList:
            self.dict[i] = np.random.dirichlet(np.ones(groupNum))

    def getUserProbByGroup(self, userId, groupId):
        """
        Get the probability that the user belongs to this group
        """
        return self.dict[userId].tolist()[groupId]

    def getProbByUserId(self, userId):
        """
        Args:
            return group prob list of the input user
        """
        return self.dict[userId].tolist()


    def getUserByGroup(self, groupId):
        """
        Args:
            return userID in the input group, group ID starts from 0
        """
        groupList=[]
        for i in self.userList:
            userProb = self.dict[i].tolist()
            if userProb.index(max(userProb)) == groupId:
                groupList.append(i)
        return groupList
