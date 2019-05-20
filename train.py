import hmm as hmm
import pandas as pd
import numpy as np
from group import Group
from membership import MembershipVector
from trajectory import Trajectory
from hmmlearn.utils import iter_from_X_lengths
import multiprocessing as mp
from functools import partial
import datetime
import time


N_STATES = 10
GROUP_NUM = 10

def main():
	trajectorydata = pd.read_csv("./trainTrajectory.csv")
	member = MembershipVector(trajectorydata['UserID'].unique(), GROUP_NUM)
	t = Trajectory(trajectorydata)

	models = [hmm.GroupLevelHMM(n_components=N_STATES, init_params='mce')
													 for i in range(GROUP_NUM)]

	for n in range(10):
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

def eval_group_hmms(membership, models):
	trajectorydata = pd.read_csv("./testTrajectory_final.csv")
	t = Trajectory(trajectorydata)
	data, length, prob_list = t.getDataWithAllGroups(membership)
	index = 0
	avg_prob = 0
	for i, j in iter_from_X_lengths(data, length):
		prob_sum = 0
		for g in range(0, GROUP_NUM):
			prob_sum += np.exp(models[g].score(data[i:j])) * prob_list[index][g]
		avg_prob += prob_sum / GROUP_NUM
		index += 1
	return np.log(avg_prob / len(length))

def train_model_for_group(groupId, models, member, t):
	data, length, proba = t.getData(groupId, member)
	models[groupId].set_weights(proba)
	models[groupId].fit(data, length)
	print(str(groupId) + "th group done")

def update_group(group):
	group.update()

def main_multiprocess():

	trajectorydata = pd.read_csv("./trainTrajectory_smaller.csv")
	member = MembershipVector(trajectorydata['UserID'].unique(), GROUP_NUM)
	t = Trajectory(trajectorydata)

	models = [hmm.GroupLevelHMM(n_components=N_STATES, init_params='mce')
													 for i in range(GROUP_NUM)]
	log = open('./logs/log_' + str(datetime.datetime.now()) + '.txt', 'w')
	for n in range(30):
		print("STAGE : " + str(n+1))
		p = mp.Pool(processes=mp.cpu_count()-1)
		# iterate groups
		start = time.time()
		manager = mp.Manager()
		model_list = manager.list(models)
		processes = []
		prod_x=partial(train_model_for_group, models=models, member=member, t=t)
		model_list = p.map(prod_x, range(0, GROUP_NUM) )
		p.close() 
		p.join() 
		
		print("Training complete")
		# Grouping and update
		models = list(model_list)
		group_list = []
		for i in range(0, GROUP_NUM):
			group_list.append(Group(hmm=models[i], membership=member, trajectory=t, groupId=i))
		manager = mp.manager()
		m_group_list = manager.list(group_list)
		p = mp.Pool(processes=mp.cpu_count()-1)
		m_group_list = p.map(update_group, m_group_list)
		p.close()
		p.join()

		print("Complete")
		end = time.time()
		print('total time (s)= ' + str(end-start))

		groups = np.zeros(GROUP_NUM)
		for i in trajectorydata['UserID']:
			groups[member.getProbOfUser(i).argmax()] += 1
		print(groups)
		eval_log = eval_group_hmms(member, models)
		print(eval_log)
		log = open('./logs/log_' + str(datetime.datetime.now()) + '.txt', 'w')
		log.write(str(eval_log))
		log.write(str(groups))        
		log.close()   
		
		for i in range(0, GROUP_NUM):
			output = open('./models/model_iter_'+str(n)+'_model_'+ str(i)+ '_' + str(datetime.datetime.now()) + '.pkl', 'wb')   
			s = pickle.dump(models[i], output)
			output.close()


def main_test():
	trajectorydata = pd.read_csv("./trainTrajectory_smaller.csv")
	member = MembershipVector(trajectorydata['UserID'].unique(), GROUP_NUM)
	t = Trajectory(trajectorydata)

	models = [hmm.GroupLevelHMM(n_components=N_STATES, init_params='mce')
													 for i in range(GROUP_NUM)]
	log = open('./logs/log_' + str(datetime.datetime.now()) + '.txt', 'w')
	for n in range(30):
		
		print("STAGE : " + str(n+1))
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

		groups = np.zeros(GROUP_NUM)
		for i in trajectorydata['UserID']:
			groups[member.getProbOfUser(i).argmax()] += 1
		print(groups)
		eval_log = eval_group_hmms(member, models)
		print(eval_log)
		log.write(str(eval_log) + "\n")
		log.write(str(groups) + "\n\n")        
		log.close()   
		
		for i in range(0, GROUP_NUM):
			output = open('./models/model_iter_'+str(n)+'_model_'+ str(i)+ '_' + str(datetime.datetime.now()) + '.pkl', 'wb')   
			s = pickle.dump(models[i], output)
			output.close()



if __name__ == '__main__':
	main_multiprocess()

