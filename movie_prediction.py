#CS 412 Final Project (Movie Ratings Prediction: Kaggle)

import numpy as np
#import numpy.linalg as la
#import scipy as sp
import math as m
#import matplotlib.pyplot as plt
#from operator import itemgetter as ig
# import pandas as pd
# import tensorflow as tf
# from sklearn.naive_bayes import GaussianNB

data_movie = open('./Data/movie.txt').readlines()
#data_sample = open('/Data/sample.txt').readlines()
data_test = open('./Data/test.txt').readlines()
#data_train = open('./Data/train.txt').readlines()
data_user = open('./Data/user.txt').readlines()

# genres = {}

data_movie.pop(0)
data_user.pop(0)
#data_train.pop(0)
data_test.pop(0)
movie_details = [movie.split(',') for movie in data_movie]
movies = {}
user_details = [user.split(',') for user in data_user]
users = {}

# # print (movie_details)

# # for movie in movie_details:
# # 	genre = movie[2].split('|')
# # 	for gen in genre:
# # 		gen = gen.strip('\n')
# # 		if (gen in genres):
# # 			genres[gen] += 1
# # 		else:
# # 			genres[gen] = 1

# # for gen, val in sorted(genres.items(), key = ig(1), reverse = True):
# # 	print (gen, ": ", val)


for movie in movie_details:
	ID = int(movie[0])
	year = movie[1]
	gen = movie[2].split('|')
	for i, g in enumerate(gen):
		gen[i] = gen[i].strip('\n')
	movies[ID] = (year, gen)

#print (movies[132])

# users = {}

for user in user_details:
	ID = int(user[0])
	gender = user[1]
	age = int(user[2]) if user[2] != 'N/A' else 'N/A'
	occupation = int(user[3]) if user[3] != 'N/A' and user[3] != 'N/A\n' else 'N/A'
	users[ID] = (gender, age, occupation)

#print (users[192])

# #Code to create the new_training data set.
# new_training_set = []

# for entry in data_train:
# 	entry = entry.split(',')
# 	ID = entry[0]
# 	user_id = int(entry[1])
# 	movie_id = int(entry[2])
# 	rating = int(entry[3].strip('\n'))

# 	new_entry = (users[user_id], movies[movie_id], rating)
# 	new_training_set.append(new_entry)

# #print (new_training_set[60])

# new_file = open('new_train.txt', 'a')
# new_file.write('Gender, Age, Occupation, Year_Movie_Was_Released, Genre, Rating_Given \n')
# for entry in new_training_set:
# 	gen = '/ '.join(str(e) for e in entry[1][1])
# 	to_write = str(entry[0][0]) + ', ' + str(entry[0][1]) + ', ' + str(entry[0][2]) + ', ' + str(entry[1][0]) + ', ' + str(gen) + ', ' + str(entry[2]) + '\n'
# 	new_file.write(to_write)





updated_training_set = open('new_train.csv', 'r')
#test_data_set = open('test.csv', 'r')

#COLUMNS = ['Gender', 'Age', 'Occupation', 'Year', 'Genre', 'Rating']

# df_train = pd.read_csv(updated_training_set, names = COLUMNS, skipinitialspace = True)
# df_test = pd.read_csv(test_data_set, names = COLUMNS, skipinitialspace = True, skiprows = 1)

x = updated_training_set.readlines()
x.pop(0)
train = []

total_count = 0
# count_male = 0
# count_female = 0
#ages = []


for entry in x:
	total_count += 1
	entry = entry.split(',')
	for i, k in enumerate(entry):
		entry[i] = entry[i].strip(' ')
	gender = entry[0]
	age = int(entry[1]) if entry[1] != 'N/A' else 'N/A'
	occupation = int(entry[2]) if entry[2] != 'N/A' else 'N/A'
	year = int(entry[3]) if entry[3] != 'N/A' else 'N/A'
	genres = set(entry[4].split('/'))
	rating = int(entry[5])
	train.append((gender, age, occupation, year, genres, rating))


predictions = open('predicted_ratings.txt', 'w')
predictions.write('Id,rating\n')
####### NAIVE BAYESIAN #########
i = 0
for sample_test in data_test:
	i += 1
	#sample_test = data_test[15]
	sample_test = sample_test.split(',')
	test_id = int(sample_test[0])
	test_user_id = int(sample_test[1])
	test_movie_id = int(sample_test[2])
	test_user_attributes = users[test_user_id]
	test_user_gender = test_user_attributes[0]
	test_user_age = int(test_user_attributes[1]) if test_user_attributes[1] != 'N/A' else 'N/A'
	test_user_occupation = test_user_attributes[2]
	test_movie_attributes = movies[test_movie_id]
	test_movie_year = test_movie_attributes[0]
	test_movie_genres = set(test_movie_attributes[1])

	#print (test_movie_attributes)
	#print (test_user_attributes)

	rating_probs = []

	for rating in range(1,6):
		count_age_matches = 0
		count_gender_matches = 0
		count_occupation_matches = 0
		count_year_matches = 0
		count_genre_matches = 0
		count_rating_match = 0
		ages = []
		count_male = 0
		count_female = 0
		for item in train:
			if item[5] == rating:
				if item[1] != 'N/A': ages.append(age)
				if item[0] == 'M': count_male += 1
				if item[0] == 'F': count_female += 1
				count_rating_match += 1
				if test_user_gender != 'N/A' and item[0] != 'N/A' and test_user_gender == item[0] :
					count_gender_matches += 1
				if test_user_age != 'N/A' and item[1] != 'N/A' and test_user_age == item[1]:
					count_age_matches += 1
				if test_user_occupation != 'N/A' and item[2] != 'N/A' and test_user_occupation == item[2]:
					count_occupation_matches += 1
				if test_movie_year != 'N/A' and item[3] != 'N/A' and test_movie_year == item[3]:
					count_year_matches += 1
				if test_movie_genres & item[4] != set():
					count_genre_matches += 1
		prob_age = 0.
		prob_gender = 0.
		prob_occ = 0.
		prob_year = 0.
		prob_genre = 0.
		
		if count_gender_matches == 0:
			if (count_male > 0 or count_female > 0):
				prob_gender = count_male if count_male > count_female else count_female
				prob_gender /= count_male + count_female
			else:
				prob_gender = 0.5
		else:
			prob_gender = count_gender_matches/count_rating_match
		age_std = np.std(ages)
		if count_age_matches == 0:
			if (len(ages) != 0 and age_std != 0):
				prob_age = 1/(np.sqrt(2*np.pi*age_std))
			else: prob_age = 0.02
		else:
			prob_age = count_age_matches/count_rating_match

		if count_occupation_matches == 0:
			prob_occ = 0.1
		else:
			prob_occ = count_occupation_matches/count_rating_match

		if count_year_matches == 0:
			prob_year = 0.02
		else:
			prob_year = count_year_matches/count_rating_match

		if count_genre_matches == 0:
			prob_genre = 0.1
		else:
			prob_genre = count_genre_matches/count_rating_match

		prob_rating = prob_age * prob_gender * prob_occ * prob_year * prob_genre * (count_rating_match/len(train))
		rating_probs.append(prob_rating)
		#print (prob_rating)

	predicted_rating = 0
	if ((rating_probs[0] + rating_probs[1] + rating_probs[2])/3 > (rating_probs[3] + rating_probs[4])/2):
		predicted_rating = np.argmax(rating_probs[:3]) + 1
	else:
		predicted_rating = np.argmax(rating_probs[3:]) + 4
	#predicted_rating = np.argmax(rating_probs) + 1
	#print (rating_probs)
	#print (predicted_rating)
	if i%1000 == 0: print("i:", i)

	prediction = str(test_id) + ',' + str(predicted_rating) + '\n'
	predictions.write(prediction)

	#if (i == 10): break
















