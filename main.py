from flask import Flask
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import request, jsonify
import pandas as pd

app = Flask(__name__)


@app.route('/<title>')
def plot_based_recommender(title):
    print("______________________"+title)
    print(type(title))
    indices = pd.read_csv("indices.csv")
    plotDF = pd.read_csv("dataset.csv")
    tfidf = TfidfVectorizer(stop_words = 'english')
    tfidf_matrix = tfidf.fit_transform(plotDF['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    title = title.lower()
    
    try:
        idx=indices.index[indices['name']==title ].tolist()
        idx = idx[0]
        print(idx)
        print("*********************************************")
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

    
if __name__ == '__main__':
    app.run(debug = True)
