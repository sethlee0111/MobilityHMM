import numpy as np
import pandas as pd
import random
from trajectory import Trajectory
from localization import Localization
from hmm import GroupLevelHMM
import datetime
import pickle

states = 10
groupNum = 80

def main():
    venueData = pd.read_csv("./VenueID_data.csv")
    l = Localization(venueData)
    trajectorydata = pd.read_csv("./trainTrajectory_final.csv")
    #print(trajectorydata["VenueID"].describe())
    t = Trajectory(trajectorydata)
    usersgroup = l.grouping(groupNum)
    
    models = train(usersgroup=usersgroup, trajectory=t)
    for i in range(0,len(models)):
        output = open('./LocalizationModel/model'+str(i)+'.pkl', 'wb')
        s = pickle.dump(models[i], output)
        output.close()

    eval_loc_model(models=models, usersgroup=usersgroup)
        
def train(usersgroup, trajectory):
    models = []
    for users in usersgroup:
        print(str(datetime.datetime.now()) + "   User "+ str(users) + " is training HMM")
        model = GroupLevelHMM(n_components=states, init_params='mce')
        data, length, proba, dic = trajectory.getDataByUserGroup(users)
        # print(data)
        # print(length)

        model.set_weights(proba)
        models.append(model)
        model.fit(data, length)
    
    return models

def eval_loc_model(models, usersgroup):
    testData = pd.read_csv("./testTrajectory_final.csv")
    testTrajectory = Trajectory(testData)
    for i in range(0, len(models)):
        print(str(datetime.datetime.now()) + "   eval model "+ str(i))
        data, length, prob, dic = testTrajectory.getDataByUserGroup(usersgroup[i])
        print(models[i].score(data,length)/len(length))


if __name__=='__main__':
    main()
