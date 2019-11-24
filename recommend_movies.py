import csv
import numpy as np
import heapq

def create_movie_names():
	movies = dict()
	with open('ml-1m/movies.dat', 'r', encoding='latin-1') as datafile:
			dataReader = csv.DictReader(datafile, delimiter=':')
			for row in dataReader:
				row['id'] = int(row['id'])
				movies[row['id']] = row['movie']
	return movies

with open('baseline_collaborative_filtering.csv','r') as datafile:
    data_iter = csv.reader(datafile, delimiter = ',')
    data = [data for data in data_iter]
ratings_array = np.asarray(data, dtype = np.float32)

print('Enter user number: ', end='')
userID = int(input())
print("recommendations for user ", userID, ":")
userID -= 1

movies = heapq.nlargest(10, range(ratings_array.shape[1]), ratings_array[userID].__getitem__)
all_movies = create_movie_names()
for i in range(0, len(movies)):
	movies[i] = movies[i] + 1
	print(i, " ", all_movies[movies[i]])

