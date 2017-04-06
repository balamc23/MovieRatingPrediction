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

#genres = {}

data_movie.pop(0)
data_user.pop(0)
#data_train.pop(0)
data_test.pop(0)
movie_details = [movie.split(',') for movie in data_movie]
movies = {}
user_details = [user.split(',') for user in data_user]
users = {}

# # print (movie_details)

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
	genres = []
	if entry[4] != 'N/A': 
		genres = entry[4].split('/ ')
	rating = int(entry[5])
	train.append((gender, age, occupation, year, genres, rating))


predictions = open('predicted_ratings.txt', 'w')
predictions.write('Id,rating\n')
####### NAIVE BAYESIAN #########


count_rating = np.zeros(5)
count_male = np.zeros(5)
count_female = np.zeros(5)
ages = [[] for i in range(5)]
occupations = [{} for i in range(5)]
years = [[] for i in range(5)]

genre_dict = {'Drama': 1, 'Comedy': 2, 'Thriller' : 3, 'Action' : 4, 'Romance': 5, 'Horror': 6, 'Adventure': 7, 'Sci-Fi': 8, 'Children\'s' : 9, 'Crime': 10, 'War' : 11, 'Documentary' : 12, 'Musical': 13, 'Animation': 14, 'Mystery': 15, 'Fantasy': 16, 'Western': 17, 'Film-Noir': 18}

count_genres = np.zeros((5,18))



for item in train:
	rating = int(item[5])
	rating_idx = rating - 1
	count_rating[rating_idx] += 1
	if item[1] != 'N/A':
		ages[rating_idx].append(item[1])
	if item[0] == 'M': 
		count_male[rating_idx] += 1
	if item[0] == 'F': 
		count_female[rating_idx] += 1
	if item[2] != 'N/A':
		if int(item[2]) in occupations[rating_idx]:
			occupations[rating_idx][item[2]] += 1
		else:
			occupations[rating_idx][item[2]] = 1
	if item[3] != 'N/A':
		years[rating_idx].append(int(item[3]))
	train_set_genre = item[4]
	if (train_set_genre != 'N/A'):
		for genre in train_set_genre:
			if genre != 'N/A':
				count_genres[rating_idx][genre_dict[genre]-1] += 1
i = 0
for sample_test in data_test:
	i += 1
	sample_test = sample_test.split(',')
	test_id = int(sample_test[0])
	test_user_id = int(sample_test[1])
	test_movie_id = int(sample_test[2])
	test_user_attributes = users[test_user_id]
	test_user_gender = test_user_attributes[0]
	test_user_age = int(test_user_attributes[1]) if test_user_attributes[1] != 'N/A' else 'N/A'
	test_user_occupation = int(test_user_attributes[2]) if test_user_attributes[2] != 'N/A' else 'N/A'
	test_movie_attributes = movies[test_movie_id]
	test_movie_year = int(test_movie_attributes[0]) if test_movie_attributes[0] != 'N/A' else 'N/A'
	test_movie_genres = test_movie_attributes[1]

	rating_probs = []

	for rating in range(1,6):

		prob_age = 0.
		prob_gender = 0.
		prob_occ = 0.
		prob_year = 0.
		prob_genre = 0.

		age_std = np.std(ages[rating-1])
		age_mean = np.average(ages[rating-1])
		if (age_std != 0 and age_mean != 0):
			if test_user_age != 'N/A':
				prob_age = (1/(np.sqrt(2*np.pi*age_std))) * np.exp(-(test_user_age-age_mean)**2/(2*age_std**2))
			else:
				prob_age = 1/(np.sqrt(2*np.pi*age_std))
		else:
			prob_age = 0.02

		if test_user_gender != 'N/A':
			if test_user_gender == 'M':
				prob_gender = count_male[rating-1]
			elif test_user_gender == 'F':
				prob_gender = count_female[rating-1]
			prob_gender /= count_male[rating-1] + count_female[rating-1]
		else:
			prob_gender = 0.5

		if test_user_occupation != 'N/A':
			if int(test_user_occupation) in occupations[rating-1]:
				prob_occ = occupations[rating-1][int(test_user_occupation)]/sum(occupations[rating-1].values())
			else:
				prob_occ = max(occupations[rating-1].values())/sum(occupations[rating-1].values())
		else:
			prob_occ = 0.1

		years_std = np.std(years[rating-1])
		years_mean = np.average(years[rating-1])
		if (years_std != 0 and years_mean != 0):
			if test_movie_year != 'N/A':
				prob_year = (1/(np.sqrt(2*np.pi*years_std))) * np.exp(-(test_movie_year-years_mean)**2/(2*years_std**2))
			else:
				prob_year = 1/(np.sqrt(2*np.pi*years_std))
		else:
			prob_year = 0.02

		max_genre = 0
		for genre in test_movie_genres:
			if (genre != 'N/A'):
				max_genre = max(max_genre, count_genres[rating-1][genre_dict[genre]-1])
		if test_movie_genres != 'N/A':
			prob_genre = max_genre/sum(count_genres[rating-1])
		else:
			prob_genre = max(count_genres[rating-1])/sum(count_genres[rating-1])

		prob_rating = prob_age * prob_gender * prob_occ * prob_year * prob_genre * (count_rating[rating-1]/len(train))
		rating_probs.append(prob_rating)

	predicted_rating = 0
	# if ((rating_probs[0] + rating_probs[1] + rating_probs[2])/3 > (rating_probs[3] + rating_probs[4])/2):
	# 	predicted_rating = np.argmax(rating_probs[:3]) + 1
	# else:
	# 	predicted_rating = np.argmax(rating_probs[3:]) + 4
	predicted_rating = np.argmax(rating_probs) + 1

	prediction = str(test_id) + ',' + str(predicted_rating) + '\n'
	predictions.write(prediction)

	if (1%1 == 0): print (i, prediction)

















