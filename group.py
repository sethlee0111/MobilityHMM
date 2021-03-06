import numpy as np
import pandas as pd
from hmm import GroupLevelHMM
from membership import MembershipVector
from trajectory import Trajectory

class Group():
    def __init__(self, hmm, membership, trajectory, groupId):
        self._hmm = hmm
        self._membership = membership
        self._trajectory = trajectory
        self._groupId = groupId
    
    def update(self):
        member = self._membership
        # X is a trajectory for each user
        p_g = self._membership.getMeanProbByGroup(self._groupId)
        if p_g > 1:
            raise ValueError("Probability cannot be bigger than 1")
        for userId in member.userList:
            trajectoryArray = self._trajectory.getTrajectoryByUser(userId)
            p_ugH = 0
            for trajectory in trajectoryArray:
                p_ugH += self._hmm.score(trajectory)
            p_guH = p_ugH + np.log(p_g)
            p_guH = np.exp(p_guH)
            member.setProbByGroupUser(p_guH, userId, self._groupId)

        return member