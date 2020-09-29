# coding: utf-8
## bvh parser sample script
import bvh_parser as bp
import pandas as pd
import nuumpy as np

if __name__ == "__main__":
	# parse bvh file
	bvh_parser = bp.bvh("Example1.bvh")
	# print first frame motion data
	print(bvh_parser.motions[0])
k = 0
for chunk in pd.read_csv("S:/MS A&R/4th Sem/Thesis/Berkley MHAD/SkeletalData-20200922T160342Z-001/traintrain_data.csv", chunksize=10000):
    print(k)
    k +=1
