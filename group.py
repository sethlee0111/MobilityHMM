import numpy as np
import pandas as pd
from hmm import GroupLevelHMM
from membership import MembershipVector
from trajectory import Trajectory

class Group():
    def __init__(self, hmm, membership, trajectory):
        self._hmm = hmm
        self._membership = membership
        self._trajectory = trajectory
    
    def update(self, groupId):
        member = self._membership
        # X is a trajectory for each user
        p_g = self._membership.getMeanProbByGroup(groupId)
        for userId in member.userList:
            trajectoryArray = self._trajectory.getTrajectoryByUser(userId)
            p_ugH = 1
            for trajectory in trajectoryArray:
                p_ugh = p_ugh * self._hmm._compute_log_likelihood(trajectory)
            p_guH = p_ugH * p_g
            member.setProbByGroupUser(p_guH)

        return member