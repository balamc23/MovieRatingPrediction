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
		prob_genre = 0.05
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