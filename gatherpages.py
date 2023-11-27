import requests
import openai
import pandas as pd
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv


load_dotenv(".env")

openai.api_key = os.environ.get("OPEN_AI_KEY")

application_token = os.environ.get("tmdb_key")
movie_pages_url = "https://api.themoviedb.org/3/movie/changes?page=1"
credits_url = "https://api.themoviedb.org/3/movie/movie_id/credits?language=en-US"


headers = {
    "accept": "application/json",
    "Authorization": "Bearer " + application_token
}

response = requests.get(movie_pages_url, headers=headers)
results = response.json()["results"]



df = pd.DataFrame(columns=["title", "genres", "overview", "release_date", "runtime", "review", "language", "popularity", "cast", "crew", "content to embed"])

total_page = 54
for page in range(51, total_page + 1):
    movie_pages_url = f"https://api.themoviedb.org/3/movie/changes?page={page}"
    response = requests.get(movie_pages_url, headers=headers)
    results = response.json()["results"]
    for result in results: 
        try:
            movie_id = result["id"]
            movie_info_result = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) + "?language=en-US", headers=headers)
            movie_info_response_data = movie_info_result.json()
            movie_name = movie_info_response_data["title"]
            movie_genre_list = movie_info_response_data["genres"]
            movie_genre = []
            for genre in movie_genre_list:
                movie_genre.append(genre["name"])
            movie_overview = movie_info_response_data["overview"]
            movie_release_date = movie_info_response_data["release_date"]
            movie_runtime = movie_info_response_data["runtime"]
            movie_review = movie_info_response_data["vote_average"]
            movie_language = movie_info_response_data["original_language"]
            movie_popularity = movie_info_response_data["popularity"]

            credits_info_result = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) + "/credits?language=en-US", headers=headers)
            credits_info_response_data = credits_info_result.json()
            movie_credits_cast = credits_info_response_data["cast"]
            movie_cast = []
            for cast in movie_credits_cast:
                movie_cast.append(cast["name"])
            movie_credits_crew = credits_info_response_data["crew"]
            movie_crew = []
            for crew in movie_credits_crew:
                movie_crew.append(crew["name"])


            content_to_embed = f"Movie Genre: {movie_genre} \n Movie Overview: {movie_overview} \n Movie Release Date: {movie_release_date} \n Movie Runtime: {movie_runtime} \n Movie Review: {movie_review} \n Movie Language: {movie_language} \n Movie Popularity: {movie_popularity} \n Cast: {movie_cast} \n Crew: {movie_crew}"

            df.loc[len(df.index)] = [movie_name, movie_genre, movie_overview, movie_release_date, movie_runtime ,movie_review, movie_language, movie_popularity, movie_cast, movie_crew, content_to_embed]
        except:
            print("loading...")

df.to_csv("movieinfo-updated pages51-54.csv")