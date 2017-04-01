#CS 412 Final Project (Movie Ratings Prediction: Kaggle)

import numpy as np
import numpy.linalg as la
import scipy as sp
import math as m
import matplotlib.pyplot as plt
from operator import itemgetter as ig
import pandas as pd

data_movie = open('./Data/movie.txt').readlines()
#data_sample = open('/Data/sample.txt').readlines()
#data_test = open('/Data/test.txt').readlines()
data_train = open('./Data/train.txt').readlines()
data_user = open('./Data/user.txt').readlines()

genres = {}

data_movie.pop(0)
data_user.pop(0)
data_train.pop(0)
movie_details = [movie.split(',') for movie in data_movie]
movies = {}
user_details = [user.split(',') for user in data_user]
users = {}

for movie in movie_details:
	ID = int(movie[0])
	year = movie[1]
	gen = movie[2].split('|')
	for i, g in enumerate(gen):
		gen[i] = gen[i].strip('\n')
	movies[ID] = (year, gen)

#print (movies[153])

users = {}

for user in user_details:
	ID = int(user[0])
	gender = user[1]
	age = int(user[2]) if user[2] != 'N/A' else 'N/A'
	occupation = int(user[3]) if user[3] != 'N/A' and user[3] != 'N/A\n' else 'N/A'
	users[ID] = (gender, age, occupation)

#print (users[1061])

new_training_set = []

for entry in data_train:
	entry = entry.split(',')
	ID = entry[0]
	user_id = int(entry[1])
	movie_id = int(entry[2])
	rating = int(entry[3].strip('\n'))

	new_entry = (users[user_id], movies[movie_id], rating)
	new_training_set.append(new_entry)

#print (new_training_set[4])
new_file = open('new_train.txt', 'a')
for entry in new_training_set:
	gen = ', '.join(str(e) for e in entry[1][1])
	to_write = str(entry[0][0]) + ', ' + str(entry[0][1]) + ', ' + str(entry[0][2]) + ', ' + str(entry[1][0]) + ', ' + str(gen) + ', ' + str(entry[2]) + '\n'
	new_file.write(to_write)




#print (movie_details)

# for movie in movie_details:
# 	genre = movie[2].split('|')
# 	for gen in genre:
# 		gen = gen.strip('\n')
# 		if (gen in genres):
# 			genres[gen] += 1
# 		else:
# 			genres[gen] = 1

# for gen, val in sorted(genres.items(), key = ig(1), reverse = True):
# 	print (gen, ": ", val)







