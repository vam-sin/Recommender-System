import copy
import csv
import heapq
import numpy as np
import time

"""
    File to perform collaborative filtering on data with and without baseline approach
"""

NUM_MOVIES = 3952
NUM_USERS = 6040
PRINT_FREQUENCY = 100

def create_utility_matrix():
	# read from file and make utility matrix
	utility_matrix = np.zeros((NUM_USERS, NUM_MOVIES), np.float32)
	with open('ml-1m/ratings.dat', 'r') as datafile:
			dataReader = csv.DictReader(datafile, delimiter=':')
			i = 0
			for row in dataReader:
				if i%PRINT_FREQUENCY == 0:
					print("[Utility Matrix - Input] Rating# ", i)
				for key in row:
					row[key] = int(row[key])
				utility_matrix[row['UserID']-1][row['MovieID']-1] = row['Rating']
				i+=1
	print("Inputted total of ", i, " ratings from dataset")
	return utility_matrix


def rmse(arr1, arr2):
	'''
		calculate root mean square error
	'''
	return (np.square(arr1 - arr2).mean()**0.5)

def mae(arr1, arr2):
	'''
		calculates mean absolute error
	'''
	return np.abs(arr1 - arr2).mean()

def predict_rating_vanilla(user, item, neighbours):
	'''
		predict ratings of unrated movies using plain collaborative filtering
	'''
	global user_average_ratings, cosine_similarities, normalized_utility_matrix
	rating = 0
	for neighbour in neighbours:
		rating += (cosine_similarities[user][neighbour] * normalized_utility_matrix[neighbour][item])
	rating /= len(neighbours)	
	rating += user_average_ratings[user]
	return rating

def predict_rating_baseline(user, item, neighbours):
	'''
	If baseline is True, the baseline value b is calculated as:
		b = mu + bx + bi
	where:
		mu = global average
		bx = (user_average - mu)
		bi = (item_average - mu) (item -> movie)
	'''
	global user_average_ratings, movie_average_ratings, cosine_similarities, utility_matrix, global_movie_average_rating
	mu = global_movie_average_rating
	bx = user_average_ratings[user] - mu
	bi = movie_average_ratings[item] - mu
	b = mu + bx + bi

	numerator = 0
	denominator = 0

	for neighbour in neighbours:
		'''
			utility_matrix[neighbour][item] - (mu + bi + user_average_ratings[neighbour] - mu)
		'''
		value =  utility_matrix[neighbour][item] - (bi + user_average_ratings[neighbour])
		numerator += cosine_similarities[user][neighbour] * value
		denominator += cosine_similarities[user][neighbour]
	assert(type(numerator) == np.float64)
	assert(type(numerator) == type(denominator))
	if denominator < 0.000001:
		return 0
	rating = numerator/denominator
	rating += b
	return rating

def knn(user, k=2):
	'''
		Find the k-nearest neighbours of user using cosine/pearson similarities 
	'''
	global cosine_similarities
	neighbours = heapq.nlargest(k+1, range(cosine_similarities.shape[0]), cosine_similarities[user].__getitem__)
	if user in neighbours:
		# user cannot be a neighbour of themselves
		np.delete(neighbours, np.where(neighbours == user))
	else:
		np.delete(neighbours, k)
	return neighbours


def create_prediction_matrix(utility_matrix, baseline, k=2):
	'''
		create a matrix of predictions
	'''
	assert(utility_matrix.shape == (NUM_USERS, NUM_MOVIES))
	prediction_matrix = np.zeros(utility_matrix.shape, np.float32)
	for i in range(0, NUM_USERS):
		if i%PRINT_FREQUENCY == 0:
			if baseline:
				print("[Predictions - Baseline] ", end='')
			else:
				print("[Predictions - Vanilla] ", end='')
			print("Processed ", i, " users so far...")
		neighbours = knn(i, k)
		for j in range(0, NUM_MOVIES):
			if not baseline:
				prediction_matrix[i][j] = predict_rating_vanilla(i, j, neighbours)
			else:
				prediction_matrix[i][j] = predict_rating_baseline(i, j, neighbours)
	return prediction_matrix

#### EXECUTION STARTS HERE

start_time = time.process_time()
# read from dataset
utility_matrix = create_utility_matrix() 
try:
	with open('global_movie_average_rating.csv','r') as datafile:
	    data_iter = csv.reader(datafile, delimiter = ',')
	    data = [data for data in data_iter]
	global_movie_average_rating = np.asarray(data, dtype = np.float32)[0][0]

	with open('movie_average_ratings.csv','r') as datafile:
	    data_iter = csv.reader(datafile, delimiter = ',')
	    data = [data for data in data_iter]
	movie_average_ratings = np.asarray(data, dtype = np.float32).flatten()
	print(movie_average_ratings)

	with open('normalized_utility_matrix.csv','r') as datafile:
	    data_iter = csv.reader(datafile, delimiter = ',')
	    data = [data for data in data_iter]
	normalized_utility_matrix = np.asarray(data, dtype = np.float32)
	print(normalized_utility_matrix)

	with open('user_average_ratings.csv','r') as datafile:
	    data_iter = csv.reader(datafile, delimiter = ',')
	    data = [data for data in data_iter]
	user_average_ratings = np.asarray(data, dtype = np.float32).flatten()
	print(user_average_ratings)

	with open('user_similarities.csv','r') as datafile:
	    data_iter = csv.reader(datafile, delimiter = ',')
	    data = [data for data in data_iter]
	cosine_similarities = np.asarray(data, dtype = np.float32)
	print(cosine_similarities)
	del data_iter, data
except FileNotFoundError:
	print("user_similarities.csv is needed to run this file. Run preprocess.py to generate file.")
	print("Exiting...")
	exit()

read_time = time.process_time()

print("Starting vanilla filtering...")
vanilla_collaborative_filtering = create_prediction_matrix(utility_matrix, baseline = False)
vanilla_time = time.process_time()

print("Starting baseline filtering... Vanilla filtering took ", vanilla_time- read_time, " seconds")
baseline_collaborative_filtering = create_prediction_matrix(utility_matrix, baseline = True)
print(baseline_collaborative_filtering)
baseline_time = time.process_time()

print("Calculating RMSE")
rmse_vanilla_collaborative = rmse(vanilla_collaborative_filtering, utility_matrix)
mae_vanilla_collaborative = mae(vanilla_collaborative_filtering, utility_matrix)
vanilla_error_time = time.process_time()

rmse_baseline_collaborative = rmse(baseline_collaborative_filtering, utility_matrix)
mae_baseline_collaborative = mae(baseline_collaborative_filtering, utility_matrix)
baseline_error_time = time.process_time()

print("\t\t=== Collaborative Filtering ===")
print("\tRMSE: ", rmse_vanilla_collaborative)
print("\tMAE: ", mae_vanilla_collaborative)
print("\tPrediction time: ", vanilla_time - read_time)
print("\tError calc time: ", vanilla_error_time - baseline_time)

print("\t\t=== Baseline Collaborative Filtering ===")
print("\tRMSE: ", rmse_baseline_collaborative)
print("\tMAE: ", mae_baseline_collaborative)
print("\tPrediction time: ", baseline_time - vanilla_time)
print("\tError calc time: ", baseline_error_time - vanilla_error_time)

np.savetxt("vanilla_collaborative_filtering.csv", vanilla_collaborative_filtering, fmt = "%5.5f", delimiter=",")
np.savetxt("baseline_collaborative_filtering.csv", baseline_collaborative_filtering, fmt = "%5.5f", delimiter=",")
file_output_time = time.process_time()

print("\n\n\t\tFILE INPUT TIME: ", read_time - start_time)
print("\n\n\t\tFILE OUTPUT TIME: ", file_output_time - baseline_error_time)
print("\t\tTOTAL EXECUTION TIME: ", file_output_time - start_time)
