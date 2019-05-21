import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

class MembershipVector():
    def __init__(self, userList, groupNum):
        """
        randomly initilize the prob
        init the userID and group Probability to a map of user'

        Args:
            userList: a list of user ID
            groupNum: a integer
        """
        self.groupNum = groupNum
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

    def getMeanProbByGroup(self, groupId):
        """P(g) = Sum u∈U Mu(g)/|U|.
        """
        sum = 0
        for i in self.userList:
            userProb = self.dict[i].tolist()
            sum += userProb[groupId]
        return sum/self.groupNum

    def setProbByGroupUser(self, prob, userId, groupId):
        self.dict[userId][groupId] = prob
        return 

    def check_normal(self):
        for key in self.dict:
            sum_prob = self.dict[key]
            if sum_prob != 1:
                raise ValueError("The sum of the membership vector should be 1")

    def getProbOfUser(self, id):
        return self.dict[id]

    def normalize(self):
        for key in self.dict:
            if not (self.dict[key] > 0).all:
                raise ValueError("Probability is negative: " + str(self.dict[key]))
        for key in self.dict:
            self.dict[key] = np.concatenate(normalize(self.dict[key].reshape(1,-1)))

    def getMembershipGroupStructure(self):
        groups = []
        for i in range(0, self.groupNum):
            dic_group = {}
            for j in self.userList:
                dic_group[j] = self.dict[j][i]
            groups.append(dic_group)
        
        return groups

    def setMembershipByGroupStructure(self, groupStructure):
        for i in range(0,len(groupStructure)):
            for key in groupStructure[i]:
                self.dict[key][i] = groupStructure[i][key]
        return self

if __name__=='__main__':
    m = MembershipVector([1,2,3,4,5,6,7,8,9,10], 5)
    print(m.getMembershipGroupStructure())
    l = m.getMembershipGroupStructure()
    m.setMembershipByGroupStructure(l)