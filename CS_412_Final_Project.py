import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.stats as sp
from random import randint

#Import Training Data Using Pandas
training_data = pd.read_csv('./Data/train.txt')

#Assign 'rating' column to a pandas dataframe
target = training_data["rating"]

#Import the user and movie information
data_movie = pd.read_csv('./Data/movie.txt')
data_user = pd.read_csv('./Data/user.txt')

#Merge training data with movie information
training_data = pd.merge(training_data, data_movie, left_on = 'movie-Id', right_on = 'Id')

#Merge Training data with user information
training_data = pd.merge(training_data, data_user, left_on='user-Id', right_on='ID')

#Drop irrelevant columns
training_data = training_data.drop(['user-Id', 'movie-Id', 'Id_x', 'Id_y', 'ID'], 1)

#Rearrange columns
training_data = training_data[['Gender', 'Age', 'Occupation', 'Genre', 'Year', 'rating']]

#Assign target based on new arrangement
target = training_data["rating"]

#Function to assign Female as 1, Male as 0 and Nan as np.nan
def assignGender(c):
    if not isinstance(c['Gender'], str):
        return np.nan
    else:
        if c['Gender'] == 'F':
            return 1
        else: 
            return 0

#Applying the assignGender function to the row column, converting all 'F' with 1, 'M' with 0 and Nan with np.nan
training_data['Gender'] = training_data.apply(assignGender, axis=1)

#List of genres
genre_dict = {'Drama': 1, 'Comedy': 2, 'Thriller' : 3, 'Action' : 4, 'Romance': 5, 'Horror': 6, 'Adventure': 7, 'Sci-Fi': 8, 'Children\'s' : 9, 'Crime': 10, 'War' : 11, 'Documentary' : 12, 'Musical': 13, 'Animation': 14, 'Mystery': 15, 'Fantasy': 16, 'Western': 17, 'Film-Noir': 18}
genre_dict = list(genre_dict.keys())

#Column for each genre. 1 means the movie is of the genre, 0 means it isn't. Initialized as -1
for j in genre_dict:
    training_data[str(j)] = -1

#Function to assign the several Genre columns.
def assignGenres(row):
    if not isinstance(row['Genre'], str):
        for j in genre_dict:
            row[str(j)] = np.nan
    else:
        y = row['Genre'].split('|')
        for j in genre_dict:
            if j in y:
                row[str(j)] = 1
            else: 
                row[str(j)] = 0
    return row

#Applying the assignGenres function to each column
training_data = training_data.apply(lambda row: assignGenres(row), axis = 1)

#Dropping the original genre column, since that information is covered by the new genre specific columns
training_data = training_data.drop('Genre', 1)
target = training_data['rating']

#Getting the mean and standard deviation of the Age column
meanAge = int(training_data['Age'].mean())
stdAge = int(training_data['Age'].std())

#Function to fill missing age entries with random values between mean+-std
def fillNanAge(c):
    if c['Age'] == np.nan or c['Age'] == 'NaN':
        return randint(int(meanAge-stdAge),int(meanAge+stdAge))
    return c['Age']

#Fill the missing values in age column with a random value between mean+-std using the fillNanAge function
training_data['Age'].fillna(training_data.groupby('rating')['Age'].transform(lambda x: (randint(meanAge-stdAge,meanAge+stdAge))), inplace=True)

#Drop the rating field
training_data = training_data.drop('rating', 1)

#Filling in other missing values using heuristics based on previous data analysis
training_data['Occupation'].fillna(training_data.groupby('Age')['Occupation'].transform(lambda x: sp.mode(x)), inplace=True)
training_data['Year'].fillna(training_data['Year'].median(), inplace = True)
for i in genre_dict:
    training_data[i].fillna(training_data.groupby('Age')[i].transform(lambda x: 1 if x.mean() > 0.055 else 0), inplace=True)
training_data['Gender'].fillna(training_data.groupby('Occupation')['Gender'].transform(lambda x: 1 if x.mean() >= 0.5 else 0), inplace=True)

#Rearranging the columns in a fixed order
training_data = training_data[['Gender', 'Age', 'Occupation', 'Year', 'War', 'Mystery', 'Fantasy', 'Musical', 'Crime', 'Adventure', 'Sci-Fi',
       'Drama', 'Action', 'Documentary', 'Romance', 'Comedy', "Children's",
       'Thriller', 'Western', 'Film-Noir', 'Horror', 'Animation']]


print ("Training Data Tasks Done.")

#Read the test data and merge it with the movie and user information
test_data = pd.read_csv('./Data/test.txt')
test_data = pd.merge(test_data, data_movie, left_on = 'movie-Id', right_on = 'Id')
test_data = pd.merge(test_data, data_user, left_on='user-Id', right_on='ID')
print ("Test Data imported and merged")

#Assigning individual rows for each genre just like the training data
for j in genre_dict:
    test_data[str(j)] = -1
test_data = test_data.apply(lambda row: assignGenres(row), axis = 1)
test_data = test_data.drop(['Genre', 'ID', 'Id_y', 'user-Id', 'movie-Id'], 1)

#Assigning 1 to females, 0 to males and np.nan to Nan just like the training data
test_data['Gender'] = test_data.apply(assignGender, axis=1)

#Randomly filling the missing age entries using same heuristic as earlier
test_data['Age'].fillna(test_data.groupby('Id_x')['Age'].transform(lambda x: (randint(meanAge-stdAge,meanAge+stdAge))), inplace=True)

#Filling other missing entries using the same heuristics as used for training data
test_data['Occupation'].fillna(test_data.groupby('Age')['Occupation'].transform(lambda x: sp.mode(x)), inplace=True)
test_data['Year'].fillna(test_data['Year'].median(), inplace = True)
for i in genre_dict:
    test_data[i].fillna(test_data.groupby('Age')[i].transform(lambda x: 1 if x.mean() > 0.055 else 0), inplace=True)
test_data['Gender'].fillna(test_data.groupby('Occupation')['Gender'].transform(lambda x: 1 if x.mean() >= 0.5 else 0), inplace=True)
test_ids = test_data['Id_x']

#Rearranging columns to a fixed order
test_data = test_data[['Gender', 'Age', 'Occupation', 'Year', 'War', 'Mystery', 'Fantasy', 'Musical', 'Crime', 'Adventure', 'Sci-Fi',
       'Drama', 'Action', 'Documentary', 'Romance', 'Comedy', "Children's",
       'Thriller', 'Western', 'Film-Noir', 'Horror', 'Animation']]

print ("Test Data tasks done")

#Gaussian Naive Bayesian Classifier
def GNB(train, target, test):
    train = pd.concat([train, target], axis=1) #concatenate training data with corresponding ratings
    #Variables to hold the priors
    n_feat = train.shape[1]
    n_samp = train.shape[0]
    count_rating = np.zeros(5)
    count_male = len(train[train['Gender'] == 0])
    count_female = n_samp - count_male
    genre_dict = {'Drama': 1, 'Comedy': 2, 'Thriller' : 3, 'Action' : 4, 'Romance': 5, 'Horror': 6, 'Adventure': 7, 'Sci-Fi': 8, 'Children\'s' : 9, 'Crime': 10, 'War' : 11, 'Documentary' : 12, 'Musical': 13, 'Animation': 14, 'Mystery': 15, 'Fantasy': 16, 'Western': 17, 'Film-Noir': 18}
    genre_dict = list(genre_dict.keys())
    count_genres = np.zeros((5,18))
    count_male_r = np.zeros(5)
    count_female_r = np.zeros(5)
    year_mean_r = np.zeros(5)
    year_var_r = np.zeros(5)
    occupations = list(train.Occupation.unique())
    count_occup_r = np.zeros((5, len(occupations)))
    age_mean_r = np.zeros(5)
    age_var_r = np.zeros(5)
    data_means = train.groupby('rating').mean()
    data_variance = train.groupby('rating').var()
    
    #Calculation of Priors using the training data set
    for rating in range(1,6):
        
        count_rating[rating-1] = train['rating'][train['rating'] == rating].count()
        count_male_r[rating-1] = train[(train['rating'] == rating) & (train['Gender'] == 0)].count()[0]
        count_female_r[rating-1] = count_rating[rating-1] - count_male_r[rating-1]
        for i, genre in enumerate(genre_dict):
            count_genres[rating-1][i] = train[(train[str(genre)] == 1) & (train['rating'] == rating)].count()[0]
        for i, occupation in enumerate(occupations):
            count_occup_r[rating-1][i] = train[(train['Occupation'] == occupations[i]) & (train['rating'] == rating)].count()[0]
    
        year_mean_r[rating-1] = data_means['Year'][data_means.index == rating].values[0]
        year_var_r[rating-1] = data_variance['Year'][data_variance.index == rating].values[0]
        age_mean_r[rating-1] = data_means['Age'][data_means.index == rating].values[0]
        age_var_r[rating-1] = data_variance['Age'][data_variance.index == rating].values[0]
    
    print ('Classifier Made, Starting Predictive Task')
    
    #Prediction using the classifier made above. Standard Naive Bayesian.
    pred = np.zeros(len(test))
    for i in range(len(test)):
        rating_probs = []
        for rating in range(1,6):
            prob_age = 0.
            prob_gender = 0.
            prob_occ = 0.
            prob_year = 0.
            prob_genre = 0.
            age_mean = age_mean_r[rating-1]
            age_var = age_var_r[rating-1]
            year_mean = year_mean_r[rating-1]
            year_var = year_var_r[rating-1]
            #Using gaussian distribution to calculate probability of age given rating
            prob_age = 1/(np.sqrt(2*np.pi*age_var)) * np.exp((-(test['Age'][i]-age_mean)**2)/(2*age_var)) if (age_mean != 0 and age_var != 0) else 0.02
            prob_gender = count_male_r[rating-1]/count_rating[rating-1] if test['Gender'][i] == 0 else count_female_r[rating-1]/count_rating[rating-1]
            if prob_gender == 0: prob_gender = 0.5
            for j, occ in enumerate(occupations):
                if occ == test['Occupation'][i]:
                    prob_occ = count_occup_r[rating-1][j]/count_rating[rating-1]
            if prob_occ == 0: prob_occ = 1/len(occupations)
            
            prob_year = 1/(np.sqrt(2*np.pi*year_var)) * np.exp((-(test['Year'][i]-year_mean)**2)/(2*year_var)) if (year_mean != 0 and year_var != 0) else 0.02
            #Heuristical way of calculating probability of genre match given rating
            num_genres = 0
            for j, genre in enumerate(genre_dict):
                if test[str(genre)][i] == 1:
                    prob_genre += (count_genres[rating-1][j])
                    num_genres += 1
            prob_genre /= (num_genres*count_rating[rating-1])
            if prob_genre == 0: prob_genre = 0.02
            prob_rating = prob_age * prob_gender * prob_occ * prob_year * prob_genre * (count_rating[rating-1]/n_samp)
            rating_probs.append(prob_rating)
        
        #pred[i] = np.argmax(rating_probs) + 1
        #Heuristic to increase result accuracy.
        if rating_probs[0] + rating_probs[1] + rating_probs[2] > rating_probs[3] + rating_probs[4]:
            pred[i] = np.argmax(rating_probs[:3]) + 1
        else:
            pred[i] = np.argmax(rating_probs[3:]) + 4
    
    return pred

#Call the GNB function to predict rating given training set
pred = GNB(training_data, target, test_data)

#Writing results to a file
predictions = open('nb_self_predictions.txt', 'w')
predictions.write('Id,rating\n')
for i in range(len(pred)):
    prediction = str(test_ids[i]) + ',' + str(int(round(pred[i]))) + '\n'
    predictions.write(prediction)

print ("Done!")

