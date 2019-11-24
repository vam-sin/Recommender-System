import csv
import pickle
import argparse
import numpy as np
import time
from functools import reduce
import os
from numpy.linalg import norm

'''
	Before running this code, make sure to add:
	
			UserID:MovieID:Rating:Timestamp
	
	as the first line of the ratings.dat file of the MovieLens DataSet. Also, use find and replace in any text editor to replace 
	all instances of double colon in the ratings.dat file with single colon. (Ctrl-F and then Ctrl-H in Sublime)  
'''

PRINT_FREQUENCY = 500
NUM_MOVIES = 3952
NUM_USERS = 6040
USE_PARTIAL_DATASET = False
USER_UPPER_LIMIT = 500

if USE_PARTIAL_DATASET:
	NUM_USERS = USER_UPPER_LIMIT

def create_utility_matrix():
	# read from file and make utility matrix
	utility_matrix = np.zeros((NUM_USERS, NUM_MOVIES), np.float32)
	with open(args.dataset, 'r') as datafile:
			dataReader = csv.DictReader(datafile, delimiter=':')
			i = 0
			for row in dataReader:
				if i%PRINT_FREQUENCY == 0:
					print("[Utility Matrix - Input] Rating# ", i)
				for key in row:
					row[key] = int(row[key])
				if USE_PARTIAL_DATASET and row['UserID'] >= USER_UPPER_LIMIT:
					break
				utility_matrix[row['UserID']-1][row['MovieID']-1] = row['Rating']
				i+=1
	print("Inputted total of ", i, " ratings from dataset")
	return utility_matrix

def moviewise_averages(utility_matrix):
	# calculate average rating for each movie and global rating average
	global_movie_average_rating = 0
	all_ratings_count = 0
	movie_average_ratings = []
	not_rated_movies = 0
	for j in range(0, NUM_MOVIES):
		if j%PRINT_FREQUENCY == 0:
			print("[Movie Wise Average] Movie #", j)
		movie_ave = 0
		count = 0
		for i in range(0, NUM_USERS):
			if utility_matrix[i][j] > 0:
				count+=1
				movie_ave += utility_matrix[i][j]
		global_movie_average_rating += movie_ave
		all_ratings_count += count
		try:
			movie_ave /= count
			movie_average_ratings.append(movie_ave)
		except ZeroDivisionError as e:
			# some movies have no ratings at all sadly
			not_rated_movies+=1
			print("NOTICE: No users have rated movie #: ", j)
			movie_average_ratings.append(0)
	print("WARN: ", not_rated_movies, " movies have no ratings at all.")
	global_movie_average_rating /= all_ratings_count
	global_movie_average_rating_array = []
	global_movie_average_rating_array.append(global_movie_average_rating) 
	return np.array(global_movie_average_rating_array), movie_average_ratings

def normalize_utility_matrix(utility_matrix):
	# normalize utility matrix by subtracting the average user rating from ratings of the users. 
	# save userwise averagee-rating in array 
	i = 0
	average = 0
	user_average_rating = []
	for row in utility_matrix:
		if not i%PRINT_FREQUENCY:
				print("[Utility Matrix - Normalizing] User# ", i)
		count = 0
		for j in range(0, len(row)):
			average += row[j]
			if row[j] != 0:
				count += 1
		average /= count
		user_average_rating.append(average)
		for j in range(0, len(row)):
			if row[j] != 0:
				row[j] -= average
		i+=1
	print("Normalized ", len(utility_matrix), " users successfully")
	return user_average_rating, utility_matrix

def cosine_similarity(a, b):
	# calculate cosine similarity between two users
	cos_sim = np.dot(a, b)/(norm(a)*norm(b))
	return cos_sim

def pearson_similarity(vec1, vec2):
	''' 
		Calculates pearson similarity between two vectors
		Let S be the set of all movies rated by both users (users represented by vec1 and vec2).
		For all elements in S, calculate and sum (r1 = rating of item in vec1, r2 = rating in vec2):
		Note: ave(vec_i) is average user-wise movie rating given by the user represented by vec_i.

			numerator = (r1 - ave(vec1)) * (r2 - ave(vec2)) -> summed over all common items
			denominator**2 = sigma((r1 - ave(vec1))**2) * sigma((r2 - ave(vec2))**2)

		pearson_similarity = numerator/denominator

		Looking at the formula, we notice that Pearson distance is merely the cosine distance between
		vec1 and vec2 for us since the average rating of users has been normalized to zero for us.

	'''
	return cosine_similarity(vec1, vec2)

def create_pearson_similarity_matrix(utility_matrix):
	# create a NUM_USERS * NUM_USERS 2D matrix containing the cosine distance between all the possible users
	pearson_distances = np.zeros((NUM_USERS, NUM_USERS), np.float32)
	for i in range(0, NUM_USERS):
		for j in range(i, NUM_USERS-1):
			# similarity between i-th and j-th user
			if i==j:
				pearson_distances[i][j] = 1
			else:
				pearson_distances[i][j] = pearson_similarity(utility_matrix[i], utility_matrix[j])
			pearson_distances[j][i] = pearson_distances[i][j]
	return pearson_distances

def create_cosine_similarity_matrix(utility_matrix):
	# create a NUM_USERS * NUM_USERS 2D matrix containing the cosine distance between all the possible users
	cosine_distances = np.zeros((NUM_USERS, NUM_USERS), np.float32)
	for i in range(0, NUM_USERS):
		if i%PRINT_FREQUENCY == 0:
			print("Processing row #", i)
		for j in range(i, NUM_USERS-1):
			# similarity between i-th and j-th user
			if i==j:
				cosine_distances[i][j] = 1
			else:
				cosine_distances[i][j] = cosine_similarity(utility_matrix[i], utility_matrix[j])
			cosine_distances[j][i] = cosine_distances[i][j]
	return cosine_distances

if __name__ == "__main__":
	start_time = time.process_time()
	parser = argparse.ArgumentParser(description='Create utility matrix (users * movies) and calculate averages')
	parser.add_argument('dataset')
	args = parser.parse_args()
	print("Attempt read from file: " + args.dataset)
	
	print("Creating utility matrix from dataset...")
	utility_matrix = create_utility_matrix()
	input_time = time.process_time()

	if not os.path.exists('global_movie_average_rating.csv'):
		global_movie_average_rating, movie_average_ratings = moviewise_averages(utility_matrix)
		print("Writing moviewise averages to file...")
		np.savetxt("movie_average_ratings.csv", movie_average_ratings, fmt = "%5.5f", delimiter=",")
		np.savetxt("global_movie_average_rating.csv", global_movie_average_rating, fmt = "%5.5f", delimiter=",")
		global_movie_average_rating = global_movie_average_rating[0]
	else:
		with open('global_movie_average_rating.csv','r') as datafile:
		    data_iter = csv.reader(datafile, delimiter = ',')
		    data = [data for data in data_iter]
		data_array = np.asarray(data, dtype = np.float32)
	movie_average_time = time.process_time()
	
	if not os.path.exists('user_average_ratings.csv'):
		print("Calculating user-wise averages and normalizing ratings by user...")
		user_average_rating, normalized_utility_matrix = normalize_utility_matrix(utility_matrix)
		np.savetxt("user_average_ratings.csv", user_average_rating, fmt = "%5.5f", delimiter=",")
		np.savetxt("normalized_utility_matrix.csv", normalized_utility_matrix, fmt = "%5.5f", delimiter=",")
	del utility_matrix
	normalized_time = time.process_time()

	if not os.path.exists("user_similarities.csv"):
		print("Finding all cosine and pearson distances.")
		cosine_distances = create_cosine_similarity_matrix(normalized_utility_matrix)
		np.savetxt("user_similarities.csv", cosine_distances, fmt = "%5.5f", delimiter=",")
	cosine_distance_time = time.process_time()

	print("============= TIME TAKEN & STATISTICS =============\n\n\n")
	print("\t\tInput time: ", input_time - start_time)
	print("\t\tCalculating movie-wise averages: ", movie_average_time - input_time)
	print("\t\tNormalizing time: ", normalized_time - movie_average_time)
	print("\t\tUser cosine differences matrix: ", cosine_distance_time - normalized_time)
	print("\t\tTOTAL TIME: ", cosine_distance_time - start_time)
