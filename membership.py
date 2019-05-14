import pandas as pd
import numpy as np

class MembershipVector():
    """
    userList: a list of user ID
    groupNum: a integer
    """
    def __init__(self, userList, groupNum):
        # randomly initilize the proba
        # init the userNumber and groupNumber to an map of user
        self.member = map()
        for userId in userList:
            self.member.append(userId, nparray)

    def getProbByUserId(self, userId):
        return

    """ return the maximum probablity that a group contains which user
    """
    def getUserGroupProb(self, groupId):
        return
