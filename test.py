# import requests
# import openai
# import pandas as pd
# import numpy as np
# import ast

# openai.api_key = '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'

# application_token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNTA3YzMyNTczMjZmYjliMGZhOWQ0NmZhZjRjMTBkZCIsInN1YiI6IjY1NDZhYzQxMjg2NmZhMDBjNDI0MDRmMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.bDOlrA_ZPVUKQ7asRdM_VyXg1O6IwmGbpn4GmB0FzoE"
# movie_pages_url = "https://api.themoviedb.org/3/movie/changes?page=1"
# credits_url = "https://api.themoviedb.org/3/movie/movie_id/credits?language=en-US"

# headers = {
#     "accept": "application/json",
#     "Authorization": "Bearer " + application_token
# }

# response = requests.get(movie_pages_url, headers=headers)
# results = response.json()["results"]

# df = pd.DataFrame(columns=["title", "genres", "overview", "release_date", "runtime", "review", "content to embed"])


# total_pages = 5
# for page in range(1, total_pages + 1):
#     movie_pages_url = f"https://api.themoviedb.org/3/movie/changes?page={page}"
#     response = requests.get(movie_pages_url, headers=headers)
#     results = response.json()["results"]
#     for result in results:
#         try:
#             movie_id = result["id"]
#             movie_info_result = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) + "?language=en-US", headers=headers)
#             movie_info_response_data = movie_info_result.json()
#             movie_name = movie_info_response_data["title"]
#             movie_genre_list = movie_info_response_data["genres"]
#             movie_genre = []
#             for genre in movie_genre_list:
#                 movie_genre.append(genre["name"])

#             movie_overview = movie_info_response_data["overview"]
#             movie_release_date = movie_info_response_data["release_date"]
#             movie_runtime = movie_info_response_data["runtime"]
#             movie_review = movie_info_response_data["vote_average"]
#             content_to_embed = f"Movie Name: {movie_name} \n Movie Genre: {movie_genre} \n Movie Overview: {movie_overview} \n Movie Release Date: {movie_release_date} \n Movie Runtime: {movie_runtime} \n Movie Review: {movie_review}"
#             df.loc[len(df.index)] = [movie_name, movie_genre, movie_overview, movie_release_date, movie_runtime ,movie_review, content_to_embed]
#         except:
#             print ("loading...")


# # for result in results: 
# #     try:
# #         movie_id = result["id"]
# #         movie_info_result = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) + "?language=en-US", headers=headers)
# #         movie_info_response_data = movie_info_result.json()

# #         movie_name = movie_info_response_data["title"]
# #         movie_genre_list = movie_info_response_data["genres"]
# #         movie_genre = []
# #         for genre in movie_genre_list:
# #             movie_genre.append(genre["name"])

# #         movie_overview = movie_info_response_data["overview"]
# #         movie_release_date = movie_info_response_data["release_date"]
# #         movie_runtime = movie_info_response_data["runtime"]
# #         movie_review = movie_info_response_data["vote_average"]
# #         content_to_embed = f"Movie Name: {movie_name} \n Movie Genre: {movie_genre} \n Movie Overview: {movie_overview} \n Movie Release Date: {movie_release_date} \n Movie Runtime: {movie_runtime} \n Movie Review: {movie_review}"
# #         df.loc[len(df.index)] = [movie_name, movie_genre, movie_overview, movie_release_date, movie_runtime ,movie_review, content_to_embed]
# #     except:
# #         print("uh oh sphagetti o's, something's wrong!")


# df.to_csv("movieinfo.csv")
# movies_df = pd.read_csv("movieinfo.csv")

# from openai.embeddings_utils import get_embedding

# def get_embeddings_batch(texts, engine):
#     embeddings = []
#     BATCH_SIZE = 2048  
#     for i in range(0, len(texts), BATCH_SIZE):
#         batch = texts[i:i + BATCH_SIZE]
#         try:
#             response = openai.Embedding.create(input=batch, engine=engine)
#             batch_embeddings = [item["embedding"] for item in response["data"]]
#             embeddings.extend(batch_embeddings)
#         except openai.error.OpenAIError as e:
#             print(f"An error occurred: {e}")
#     return embeddings


# texts_to_embed = movies_df["content to embed"].tolist()
# movies_df_embeddings = get_embeddings_batch(texts_to_embed, engine='text-embedding-ada-002')
# movies_df['embedding'] = movies_df_embeddings
# movies_df.to_csv('movieinfo-embeddings.csv')


# # movies_df['embedding'] = movies_df["content to embed"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# # movies_df.to_csv('movieinfo-embeddings.csv')

# movies_df = pd.read_csv('movieinfo-embeddings.csv')
# movies_df['embedding'] = movies_df['embedding'].apply(ast.literal_eval).apply(np.array)

# user_input = input('Tell me any vibe, genre, emotion, etc. you\'re feeling, and I\'ll recommend a movie: ')

# user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")


# from openai.embeddings_utils import cosine_similarity

# movies_df["similarities"] = movies_df['embedding'].apply(lambda x: cosine_similarity(x, user_input_vector))
# recommended_movies = movies_df.sort_values("similarities", ascending=False).head(20)['title']

# print("Movies recommended for you: ")
# for title in recommended_movies:
#     print(title)



import pandas as pd
import numpy as np
import openai
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv(".env")

openai.api_key = os.environ.get("OPEN_AI_KEY")


movies_df = pd.read_csv("masterdf-updated.csv")
movies_df['embedding'] = movies_df['embedding'].apply(eval).apply(np.array)

user_input = input('Tell me any vibe, genre, emotion, etc. you\'re feeling, and I\'ll recommend a movie: ')

user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")

import time

try:
    user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")
except openai.error.RateLimitError as e:
    print(f"Loading...")
    time.sleep(20)
    user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")



from sklearn.metrics.pairwise import cosine_similarity

movies_df["similarities"] = movies_df['embedding'].apply(
    lambda x: cosine_similarity([x], [user_input_vector])[0][0]
    )
recommended_movies = movies_df.sort_values("similarities", ascending=False).head(100)
sorted_recommended_movies = recommended_movies.sort_values("popularity", ascending=False).head(10)
top_20_movies = sorted_recommended_movies['title']





print("Movies recommended for you: ")
for title in top_20_movies:
    print(title)