# AirwayTestData
1. testData.7z package: 
there are a original segmentation that contains airway mask and skeleton( bifurcation point).
you could load this data in python using:
        d = np.load('.\keypoints\data\PA000019.npz', allow_pickle=True)
        # get airway mask
        mask = d['isoMask']
        # get skeleton of the whole airway tree
        lines = d['skeleton'] 
        # for example, you could extrac the first line of skeletion
        line = lines[0]
        # the information of the first line: 
        startpoint = line['endpoint1'] # the starting point of the first line
        endpoint = line['endpoint2'] # the ending point of the first line
        flag1 = line['p1IsBif'] # False, it means the starting point is not a bifurcation point
        flag2 = line['p2IsBif'] # True, it means the ending point is a bifurcation point
        
vmtkScript.py: The python script that exctract center lines from airway mask using VMTK
