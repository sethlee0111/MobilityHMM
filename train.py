import hmm as hmm
import pandas as pd
import numpy as np
from group import Group
from membership import MembershipVector
from trajectory import Trajectory

N_STATES = 10
GROUP_NUM = 10

def main():

	trajectorydata = pd.read_csv("./trainTrajectory.cvs")
	member = MembershipVector(trajectorydata['UserID'].unique(), GROUP_NUM)
	t = Trajectory(trajectorydata)

	models = [hmm.GroupLevelHMM(n_components=N_STATES, init_params='mce')
													 for i in range(GROUP_NUM)]

	for n in range(3):
		print("STAGE : " + n)
		# iterate through groups
		for i in range(1, GROUP_NUM+1):
			print("LEARNING FOR GROUP " + i)
			data, length, proba = t.getData(i, member)
			models[i].set_weights(proba)
			models[i].fit(data, length)

		print("Grouping...")
		# Grouping and update
		for i in range(1, GROUP_NUM+1):
			g = Group(hmm=models[i], membership=member, trajectory=t, groupId=i)
			member = g.update()

	#t.getTrajectoryByUser(1)
	t.getBaseModelData()

if __name__ == '__main__':
	main()
