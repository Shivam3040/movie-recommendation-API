from flask import Flask, jsonify, request 
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_restful import Resource, Api 
import pandas as pd
# creating the flask app 
app = Flask(__name__) 
# creating an API object 
api = Api(app) 

# making a class for a particular resource 
# the get, post methods correspond to get and post requests 
# they are automatically mapped by flask_restful. 
# other methods include put, delete, etc. 
class Hello(Resource): 

	# corresponds to the GET request. 
	# this function is called whenever there 
	# is a GET request for this resource 
	def get(self,title):
		indices = pd.read_csv("indices.csv")
		plotDF = pd.read_csv("dataset.csv")
		tfidf = TfidfVectorizer(stop_words = 'english')
		tfidf_matrix = tfidf.fit_transform(plotDF['description'])
		cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
		title = title.lower()
		try:
			idx=indices.index[indices['name']==title ].tolist()
			idx = idx[0]
		except KeyError:
			print('Movie does not exist :(')
			return jsonify({"key":False})
		sim_scores = list(enumerate(cosine_sim[idx]))
		sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
		sim_scores = sim_scores[1:11]
		movie_indices = [sim_score[0] for sim_score in sim_scores]
		df = indices['name'].iloc[movie_indices]
		df = df.to_dict()
		return jsonify(df)
	# Corresponds to POST request 
	def post(self,title):
		indices = pd.read_csv("indices.csv")
		plotDF = pd.read_csv("dataset.csv")
		tfidf = TfidfVectorizer(stop_words = 'english')
		tfidf_matrix = tfidf.fit_transform(plotDF['description'])
		cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
		title = title.lower()
		try:
			idx=indices.index[indices['name']==title ].tolist()
			idx = idx[0]
		except KeyError:
			print('Movie does not exist :(')
			return jsonify({"key":False})
		sim_scores = list(enumerate(cosine_sim[idx]))
		sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
		sim_scores = sim_scores[1:11]
		movie_indices = [sim_score[0] for sim_score in sim_scores]
		df = indices['name'].iloc[movie_indices]
		df = df.to_dict()
		return jsonify(df)



api.add_resource(Hello, '/<string:title>') 



# driver function 
if __name__ == '__main__': 

	app.run(debug = True) 









