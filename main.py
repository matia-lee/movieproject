import pandas as pd
import numpy as np
import openai
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv(".env")

openai.api_key = os.environ.get("OPEN_AI_KEY")

movies_df = pd.read_csv("./master_data/masterdf-updated-notitle.csv")
movies_df['embedding'] = movies_df['embedding'].apply(eval).apply(np.array)

user_input = input('What type of movie would you like to see? ')

user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")


from openai.embeddings_utils import cosine_similarity



popularity_threshold = 13



movies_df["similarities"] = movies_df['embedding'].apply(lambda x: cosine_similarity(x, user_input_vector))
filtered_movies_df = movies_df[movies_df['popularity'] >= popularity_threshold]
recommended_movies = filtered_movies_df.sort_values("similarities", ascending=False).head(10)
top_10_movies = recommended_movies['title']

print("Movies recommended for you: ")
for title in top_10_movies:
    print (title)


