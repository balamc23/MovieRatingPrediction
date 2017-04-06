#combine predictions
file_name = 'predicted_ratings_'
filenames = [file_name + str(i) + '.txt' for i in range(33)]
with open('predicted_ratings_combined.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)