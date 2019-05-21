import numpy as np
import pandas as pd
import random
from trajectory import Trajectory
from localization import Localization
from hmm import GroupLevelHMM
import datetime
import pickle

states = 10
groupNum = 10

def main():
    venueData = pd.read_csv("./VenueID_data.csv")
    l = Localization(venueData)
    trajectorydata = pd.read_csv("./trainTrajectory_smaller.csv")
    #print(trajectorydata["VenueID"].describe())
    t = Trajectory(trajectorydata)
    usersgroup = l.grouping(groupNum)

    testtrajectorydata = pd.read_csv("./testTrajectory_smaller.csv") 
    testTrajectory = Trajectory(testtrajectorydata)


    models, dics = train(usersgroup=usersgroup, trajectory=t)
    for i in range(0,len(models)):
        output = open('./LocalizationModel/'+str(groupNum)+'model_state'+str(states)+"_"+str(i)+'.pkl', 'wb')
        s = pickle.dump(models[i], output)
        output.close()

        eval_loc_model(testTrajectory=testTrajectory, model=models[i], users=usersgroup[i], dic=dics[i])
        
def train(usersgroup, trajectory):
    models = []
    dics = []
    for users in usersgroup:
        print(str(datetime.datetime.now()) + "   User "+ str(users) + " is training HMM")
        model = GroupLevelHMM(n_components=states, init_params='mce')
        data, length, proba, dic = trajectory.getDataByUserGroup(users)
        # print(data)
        # print(length)

        model.set_weights(proba)
        model.fit(data, length)
        models.append(model)
        dics.append(dic)
    
        print(dic)

    return models, dics

def load_models(models):
    # Load model
    for i in range(0,groupNum):
        input = open('./LocalizationModel/'+str(groupNum)+'model_state'+str(states)+"_"+str(i)+'.pkl', 'rb')
        model = pickle.load(input)
        models.append(model)
        input.close()
    return models

def eval_loc_model(testTrajectory, model, users, dic):
    data, length, prob = testTrajectory.getDataByUserGroupAssignCustomVenueID(users, dic)
    print(model.score(data,length)/len(length))

def loadModel_eval():
    venueData = pd.read_csv("./VenueID_data.csv")
    l = Localization(venueData)
    trajectorydata = pd.read_csv("./trainTrajectory_final.csv")
    #print(trajectorydata["VenueID"].describe())
    t = Trajectory(trajectorydata)
    usersgroup = l.grouping(groupNum)
    models = []
    models = load_models(models)
    testtrajectorydata = pd.read_csv("./testTrajectory_smaller.csv") 
    testTrajectory = Trajectory(testtrajectorydata)
    for i in range(0, len(models)):
        print(str(datetime.datetime.now()) + "   eval model "+ str(i))
        data, length, prob, dic = testTrajectory.getDataByUserGroup(usersgroup[i])
        print(models[i].score(data,length)/len(length))
    eval_loc_model(models, usersgroup)

if __name__=='__main__':
    loadModel_eval()
    # main()
