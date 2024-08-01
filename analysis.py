import numpy as np
import dtw 

def compute_alingment(pho1, pho2):
    pho1 = np.array(pho1)
    pho2 = np.array(pho2)
    alingment = dtw(pho1, pho2, keep_internals = True)
    return alingment

def analyze_alingment(alingment, pho1, pho2):
    matches = []
    mismatches = []

    for i, j in zip(alingment.index1, alingment.index2):
        if pho1[i] == pho2[j]:
            matches.append((pho1[i], pho2[j]))
        else:
            mismatches.append((pho1[i], pho2[j]))
    return matches, mismatches

def calculate_confidence(alingment):
    distance = alingment.distance 
    max_distance = max(len(alingment.index1), len(alingment.index2))
    confidence = 1 - (distance / max_distance)
    return confidence 