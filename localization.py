import numpy as np
import pandas as pd
import random

class Localization():
    """Agglomerative Hierarchical Clustering
    greedy algorithm to compute the most likely grouping
    """
    def __init__(self, df):
        self._data = df
        # self.userList = df['UserID'].unique()
        self.userList = []
        for i in range(1,80):
            self.userList.append(i)
        self.dict = {}
        for userId in self.userList:
            self.dict[userId] = df.loc[df['UserID'] == userId]['Venue category ID'].unique()

    def grouping(self, groupNum):
        random.seed(5201314)
        group = {}
        for i in self.dict.keys():
            l = str(i)
            group[l] = self.dict[i]

        num = len(list(group.keys()))

        while(num > groupNum):
            randomUser = random.choice(list(group.keys()))
            dic = {}
            for i in group.keys():
                dic[i] = self.jaccard_similarity(group[randomUser], group[i])
            dic.pop(randomUser)
            maximum = max(dic, key=dic.get)  # Just use 'min' instead of 'max' for minimum.
            # if dic[maximum] < 0.01:
            #     continue
            first_list = group.pop(maximum)
            second_list = group.pop(randomUser)

            in_first = set(first_list)
            in_second = set(second_list)

            in_second_but_not_in_first = in_second - in_first

            result = list(first_list) + list(in_second_but_not_in_first)

            sum = (maximum) + ";" + (randomUser)
            group[sum] = result
            num = len(group.keys())
            print("num size : " + str(num))
            # print(len(group.keys()))

        re = []
        for usergroup in list(group.keys()):
            strs = usergroup.split(";")
            re.append(strs)

        return re

    def jaccard_similarity(self, list1, list2):
        #print(list1)
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection / union)

def main():
    trajectorydata = pd.read_csv("./VenueID_data.csv")
    l = Localization(trajectorydata)
    re = l.grouping(80)
    print(re)
    

if __name__=='__main__':
    main()


