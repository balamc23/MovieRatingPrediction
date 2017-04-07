#combine predictions
file_name = './Results/predicted_ratings_'
filenames = [file_name + str(i) + '.txt' for i in range(33)]
#filenames = ['predicted_ratings_0.txt', 'predicted_ratings_1.txt']
with open('predicted_ratings_combined.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)