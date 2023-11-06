import requests
import openai
import pandas as pd
import numpy as np

openai.api_key = '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'

application_token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNTA3YzMyNTczMjZmYjliMGZhOWQ0NmZhZjRjMTBkZCIsInN1YiI6IjY1NDZhYzQxMjg2NmZhMDBjNDI0MDRmMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.bDOlrA_ZPVUKQ7asRdM_VyXg1O6IwmGbpn4GmB0FzoE"
movie_pages_url = "https://api.themoviedb.org/3/movie/changes?page=1"
credits_url = "https://api.themoviedb.org/3/movie/movie_id/credits?language=en-US"


# this entire section below is "authenticating" the information from the API source. 
# It makes sure through my applciation token that my request to grab information is verified. 
# If it is verified, then the information is stored in response. 
# From there, I store the information I want (Which is in the results key in the json) and store that into results
headers = {
    "accept": "application/json",
    "Authorization": "Bearer " + application_token
}

response = requests.get(movie_pages_url, headers=headers)
results = response.json()["results"]


# creates the columns for the panda dataframe. column headers for the csv file
df = pd.DataFrame(columns=["title", "genres", "overview", "release_date", "runtime", "review", "content to embed"])


# for result in results means that in the json dictionary in the movie url, the results in "results" will loop
for result in results: 
    # now it is looping everything under "id" specifically in "results", and assigns the movie id to the variable movie_id
    try:
        movie_id = result["id"]
        movie_info_result = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) + "?language=en-US", headers=headers)
        movie_info_response_data = movie_info_result.json()

        # so right now, i have made a variable: movie_info_response_data that has access to the information to SPECIFIC MOVIE IDS
        # This means that my loop is getting the values of each of the IDs on the FIRST PAGE, 
        # and then from there it is going through the specific information for each of the IDs
        movie_name = movie_info_response_data["title"]

        # okay so big sexy yash helped me with this. Basically i am doing the same thing as above, except I have a small problem
        # the problem is that under genres in the database, there is a list of dictionaries (basically more than one thing called genre)
        # so now what's happening is that, I make an empty list called movie_genre, and then create a forloop
        # the forloop is running through the dictionary "genres" in the database and it is adding anything that's labelled "name" in the dictionary to my list, movie_genre
        # then, it runs through and finds everything else in "name" in the dictionary list and adds it to the end of the list
        movie_genre_list = movie_info_response_data["genres"]
        movie_genre = []
        for genre in movie_genre_list:
            movie_genre.append(genre["name"])

        movie_overview = movie_info_response_data["overview"]
        movie_release_date = movie_info_response_data["release_date"]
        movie_runtime = movie_info_response_data["runtime"]
        movie_review = movie_info_response_data["vote_average"]

        # since the vector dot product works between comparing two things, I created this last column content_to_embed
        # the purpose of this column is to gather all the information in the previous columns into one, so that I can compare.
        # the reason why I still have the other columns is in case I need to access something specifically. 
        # it is easier to access from something that I have already defined (such as movie genres or movie titles)
        content_to_embed = f"Movie Name: {movie_name} \n Movie Genre: {movie_genre} \n Movie Overview: {movie_overview} \n Movie Release Date: {movie_release_date} \n Movie Runtime: {movie_runtime} \n Movie Review: {movie_review}"
        # for each of the movie ids, these information are going into the rows in this order. 
        # This should correspond to the order of the column headings that I defined above.
        df.loc[len(df.index)] = [movie_name, movie_genre, movie_overview, movie_release_date, movie_runtime ,movie_review, content_to_embed]
    except:
        print("loading...")

# Here, i put the information into a csv that i titled movieinfo.csv
df.to_csv("movieinfo.csv")

# In order to actually assign movieinfo.csv an output, I must read it back into a dataframe (movies_df)
movies_df = pd.read_csv("movieinfo.csv")

from openai.embeddings_utils import get_embedding

# here, i am turning the string text from "content to embed" into vectors (all the info from under this column)
# the ['embedding'] creates a new column in the movies_df and titles it embedding
# I am then embedding the string text into numbers
movies_df['embedding'] = movies_df["content to embed"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# here, I am putting those vectors into another csv, titled movieinfo-embeddings.csv. This is so i can access the vectors now instead of the text.
movies_df.to_csv('movieinfo-embeddings.csv')

# Here, since movieinfo-embeddings.csv is a string, I am converting it to a numpy array to perform operations on the strings
# this is so I can do math (dot product), which is what I actually want to do
# Again, like line 72, I need to read it back into a dataframe (movies_df) in order to manipulate the information in the csv
# Except, I am updating what movies_df represents, so it is the same variable
movies_df = pd.read_csv('movieinfo-embeddings.csv')
movies_df['embedding'] = movies_df['embedding'].apply(eval).apply(np.array)

# Here, I am prompting the user for what they want, so I can find similarities and provide a recommendation
user_input = input('Tell me any vibe, genre, emotion, etc. you\'re feeling, and I\'ll recommend a movie: ')

# Here, I am turning the user_input into a vector so that I can compare it. Basically i'm lubing it up rn (pause)
user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")


from openai.embeddings_utils import cosine_similarity

# I am creating another column called "similarities". It shows how similar the user input is to my dataframe
movies_df["similarities"] = movies_df['embedding'].apply(lambda x: cosine_similarity(x, user_input_vector))
# I am sorting the values from most similar to least (i am retrieving 20 results)
# The dictionary key 'title' at the end will hoepfully only display the title of the movies  
recommended_movies = movies_df.sort_values("similarities", ascending=False).head(20)['title']

print("Movies recommended for you: ")
for title in recommended_movies:
    print(title)