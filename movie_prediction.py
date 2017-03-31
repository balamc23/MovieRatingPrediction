#CS 412 Final Project (Movie Ratings Prediction: Kaggle)

import numpy as np
import numpy.linalg as la
import scipy as sp
import math as m
import matplotlib.pyplot as plt
from operator import itemgetter as ig

data_movie = open('./Data/movie.txt').readlines()
#data_sample = open('/Data/sample.txt').readlines()
#data_test = open('/Data/test.txt').readlines()
data_train = open('./Data/train.txt').readlines()
data_user = open('./Data/user.txt').readlines()

genres = {}

data_movie.pop(0)
movie_details = [movie.split(',') for movie in data_movie]

#print (movie_details)

for movie in movie_details:
	genre = movie[2].split('|')
	for gen in genre:
		gen = gen.strip('\n')
		if (gen in genres):
			genres[gen] += 1
		else:
			genres[gen] = 1

for gen, val in sorted(genres.items(), key = ig(1), reverse = True):
	print (gen, ": ", val)







