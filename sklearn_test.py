#CS 412 Final Project
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import Imputer
import scipy.stats as sp
from sklearn.svm import SVC

training_data = pd.read_csv('new_train.csv', sep='\s*,\s*', header=0, encoding='ascii', engine='python')
#training_data = training_data.head(50)
target = training_data["Rating_Given"]
training_data['Genre'] = training_data['Genre'].astype('category')

print ("Training Data Imported")

def assignGender(c):
    if not isinstance(c['Gender'], str):
        return np.nan
    else:
        if c['Gender'] == 'F':
            return 1
        else: 
            return 0

training_data['Gender'] = training_data.apply(assignGender, axis=1)
genre_dict = {'Drama': 1, 'Comedy': 2, 'Thriller' : 3, 'Action' : 4, 'Romance': 5, 'Horror': 6, 'Adventure': 7, 'Sci-Fi': 8, 'Children\'s' : 9, 'Crime': 10, 'War' : 11, 'Documentary' : 12, 'Musical': 13, 'Animation': 14, 'Mystery': 15, 'Fantasy': 16, 'Western': 17, 'Film-Noir': 18}
genre_dict = list(genre_dict.keys())
for j in genre_dict:
    training_data[str(j)] = -1

def assignGenres(row):
    if not isinstance(row['Genre'], str):
        for j in genre_dict:
            row[str(j)] = np.nan
    else:
        y = row['Genre'].split('/ ')
        for j in genre_dict:
            if j in y:
                row[str(j)] = 1
            else: 
                row[str(j)] = 0
    return row

training_data = training_data.apply(lambda row: assignGenres(row), axis = 1)

training_data = training_data.drop('Genre', 1)
training_data = training_data.drop('Rating_Given', 1)

training_data['Age'].fillna(training_data['Age'].mean(), inplace = True)
training_data['Occupation'].fillna(training_data.groupby('Age')['Occupation'].transform(lambda x: sp.mode(x)), inplace=True)
training_data['Year_Movie_Was_Released'].fillna(training_data['Year_Movie_Was_Released'].median(), inplace = True)
for i in genre_dict:
    training_data[i].fillna(training_data.groupby('Age')[i].transform(lambda x: 1 if x.mean() > 0.055 else 0), inplace=True)
training_data['Gender'].fillna(training_data.groupby('Occupation')['Gender'].transform(lambda x: 1 if x.mean() >= 0.5 else 0), inplace=True)

print ("Training Data Tasks Done.")


svc = SVC(kernel='linear')
svc.fit(training_data, target)

test_data = pd.read_csv('test.csv')
#test_data = test_data.head(50)
data_movie = pd.read_csv('movie.txt')
data_user = pd.read_csv('user.txt')

test_data = pd.merge(test_data, data_movie, left_on = 'movie-Id', right_on = 'Id')
test_data = pd.merge(test_data, data_user, left_on='user-Id', right_on='ID')

print ("Test Data imported and merged")

def assignGenresTest(row):
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

for j in genre_dict:
    test_data[str(j)] = -1
test_data = test_data.apply(lambda row: assignGenresTest(row), axis = 1)
test_data = test_data.drop(['Genre', 'ID', 'Id_y'], 1)

test_data['Gender'] = test_data.apply(assignGender, axis=1)
test_data['Age'].fillna(test_data['Age'].mean(), inplace = True)
test_data['Occupation'].fillna(test_data.groupby('Age')['Occupation'].transform(lambda x: sp.mode(x)), inplace=True)
test_data['Year'].fillna(test_data['Year'].median(), inplace = True)
for i in genre_dict:
    test_data[i].fillna(test_data.groupby('Age')[i].transform(lambda x: 1 if x.mean() > 0.055 else 0), inplace=True)
test_data['Gender'].fillna(test_data.groupby('Occupation')['Gender'].transform(lambda x: 1 if x.mean() >= 0.5 else 0), inplace=True)
test_data = test_data.drop(['user-Id', 'movie-Id'], 1)
test_data.rename(columns={'Year': 'Year_Movie_Was_Released'}, inplace=True)
test_ids = test_data['Id_x']
test_data = test_data[['Gender', 'Age', 'Occupation', 'Year_Movie_Was_Released', 'War', 'Mystery', 'Fantasy', 'Musical', 'Crime', 'Adventure', 'Sci-Fi',
       'Drama', 'Action', 'Documentary', 'Romance', 'Comedy', "Children's",
       'Thriller', 'Western', 'Film-Noir', 'Horror', 'Animation']]

print ("Test Data tasks done")

pred = svc.predict(test_data)
#print (pred)

print ("Predictions Done")

predictions = open('sk_learn_svm_predictions.txt', 'w')
predictions.write('Id,rating\n')
for i in range(len(pred)):
    prediction = str(test_ids[i]) + ',' + str(pred[i]) + '\n'
    predictions.write(prediction)

print ("Done!")



