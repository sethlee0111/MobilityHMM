import numpy as np
import pandas as pd
import random
from trajectory import Trajectory
from localization import Localization
from hmm import GroupLevelHMM
import datetime

states = 10

def main():
    venueData = pd.read_csv("./VenueID_data.csv")
    l = Localization(venueData)
    trajectorydata = pd.read_csv("./trainTrajectory_smaller.csv")
    t = Trajectory(trajectorydata)
    usersgroup = l.grouping(5)
    models = []
    for users in usersgroup:
        print(str(datetime.datetime.now()) + "User "+ str(users) + " is training HMM")
        model = GroupLevelHMM(n_components=states, init_params='mce')
        data, length, proba, dic = t.getDataByUserGroup(users)
        model.set_weights(proba)
        models.append(model)
        model.fit(data, length)
        


if __name__=='__main__':
    main()
