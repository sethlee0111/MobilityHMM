import hmm as hmm
import pandas as pd
import numpy as np
from group import Group
from membership import MembershipVector
from trajectory import Trajectory
from hmmlearn.utils import iter_from_X_lengths

N_STATES = 10
GROUP_NUM = 10

def main():
	trajectorydata = pd.read_csv("./trainTrajectory.csv")
	member = MembershipVector(trajectorydata['UserID'].unique(), GROUP_NUM)
	t = Trajectory(trajectorydata)

	models = [hmm.GroupLevelHMM(n_components=N_STATES, init_params='mce')
													 for i in range(GROUP_NUM)]

	for n in range(3):
		print("STAGE : " + str(n))
		# iterate through groups
		for i in range(0, GROUP_NUM):
			print("LEARNING FOR GROUP " + str(i))
			data, length, proba = t.getData(i, member)
			models[i].set_weights(proba)
			models[i].fit(data, length)

		print("Grouping...")
		# Grouping and update
		for i in range(0, GROUP_NUM):
			g = Group(hmm=models[i], membership=member, trajectory=t, groupId=i)
			member = g.update()

def eval_group_hmms(membership, models):
	trajectorydata = pd.read_csv("./testTrajectory_smaller.csv")
	t = Trajectory(trajectorydata)
	data, length, prob_list = t.getDataWithAllGroups(membership)
	index = 0
	total_score = 0
	for i, j in iter_from_X_lengths(data, length):
		score_sum = 0
		for g in range(0, GROUP_NUM):
			score_sum += models[g].score(data[i:j]) + prob_list[index][g]
		total_score += score_sum / GROUP_NUM
		index += 1
	return total_score / len(length)

def main_test():
	trajectorydata = pd.read_csv("./trainTrajectory_smaller.csv")
	member = MembershipVector(trajectorydata['UserID'].unique(), GROUP_NUM)
	t = Trajectory(trajectorydata)

	models = [hmm.GroupLevelHMM(n_components=N_STATES, init_params='mce')
													 for i in range(GROUP_NUM)]

	for n in range(3):
		print("STAGE : " + str(n))
		# iterate through groups
		for i in range(0, GROUP_NUM):
			print("LEARNING FOR GROUP " + str(i))
			data, length, proba = t.getData(i, member)
			models[i].set_weights(proba)
			models[i].fit(data, length)

		print("Grouping...")
		# Grouping and update
		for i in range(0, GROUP_NUM):
			g = Group(hmm=models[i], membership=member, trajectory=t, groupId=i)
			member = g.update()

	print(eval_group_hmms(member, models))


if __name__ == '__main__':
	main()

