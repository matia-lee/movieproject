import pandas as pd
import numpy as np
import openai
import os
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

load_dotenv(".env")

openai.api_key = os.environ.get("OPEN_AI_KEY")


# movies_df = pd.read_csv("masterdf-updated.csv")
movies_df = pd.read_csv("masterdf-updated-notitle.csv")
movies_df['embedding'] = movies_df['embedding'].apply(eval).apply(np.array)

user_input = input('What type of movie would you like to see? ')

user_input_vector = get_embedding(user_input, engine="text-embedding-ada-002")


from openai.embeddings_utils import cosine_similarity

movies_df["similarities"] = movies_df['embedding'].apply(lambda x: cosine_similarity(x, user_input_vector))
# movies_df["score"] = movies_df["similarities"] * movies_df["popularity"]
recommended_movies = movies_df.sort_values("similarities", ascending=False).head(100)
sorted_recommended_movies = recommended_movies.sort_values("popularity", ascending=False).head(10)
top_10_movies = sorted_recommended_movies['title']



print("Movies recommended for you: ")
for title in top_10_movies:
    print(title)
