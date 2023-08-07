import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/koppo/Downloads/enjoysport (1).csv")
concepts = data.iloc[:, 0:-1].values
target = data.iloc[:, -1].values

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("Initialization of specific_h and general_h")
    print(specific_h)
    print(general_h)

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
            print("Steps of Candidate Elimination Algorithm", i + 1)
            print(specific_h)
            print(general_h)
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
            print("Steps of Candidate Elimination Algorithm", i + 1)
            print(specific_h)
            print(general_h)

    # Remove inconsistent general hypotheses from the list
    general_h = [h for h in general_h if h != ['?', '?', '?', '?', '?', '?']]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_hypothesis:")
print(s_final)
print("Final General_hypothesis:")
print(g_final)
