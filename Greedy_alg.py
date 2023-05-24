import numpy as np
import random

class Greedy:
    def __init__(self, A):
        self.A = np.array(A)
        self.features = np.arange(0, self.A.shape[1])
        self.parl = np.arange(0, self.A.shape[0])
        self.result = np.zeros(self.A.shape[0])


    def Solve(self):
        parl = random.choice(self.parl)

        self.Add_parl(parl)
        while(len(self.features) != 0):
            feature_nums = np.array([])

            for i in self.parl:

                f = self.Check_features(i)
                num = self.Count_feratures(f)
                feature_nums = np.append(feature_nums, num)


            parl_num = feature_nums.argmax()

            self.Add_parl(self.parl[parl_num])

        result = np.flatnonzero(self.result == np.max(self.result))
        result = result +1

        return result




    def Check_features(self, parl):
        features_list = []
        parl_list = self.A[parl, :]
        for  i in range(0, len(parl_list)):
            if(parl_list[i] ==1):
                features_list.append(i)

        return features_list

    def Count_feratures(self, f_list):
        num = 0
        for i in f_list:
            if( i in self.features):
                num +=1

        return num


    def Add_parl( self, parl):
        features_list = self.Check_features(parl)
        mask = np.isin(self.features, features_list, invert=True)
        self.features = self.features[mask]
        self.parl = self.parl[self.parl !=parl]
        self.result[parl] = 1

