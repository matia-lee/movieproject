import openai
import pandas as pd
import time
import os
from dotenv import load_dotenv
from openai.embeddings_utils import get_embeddings

load_dotenv(".env")

openai.api_key = os.environ.get("OPEN_AI_KEY")





def get_embeddings_batch(texts, engine):
    embeddings = []
    BATCH_SIZE = 2048
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        while True:
            try:
                response = openai.Embedding.create(input=batch, engine=engine)
                batch_embeddings = [item["embedding"] for item in response["data"]]
                embeddings.extend(batch_embeddings)
                break 
            except openai.error.RateLimitError:
                print("Rate limit reached, sleeping for 20 seconds...")
                time.sleep(22)
            except openai.error.OpenAIError as e:
                print(f"An error occurred: {e}")
                break
    return embeddings



movies_df = pd.read_csv("movieinfo-updated pages51-54.csv")


texts_to_embed = movies_df["content to embed"].tolist()
movies_df_embeddings = get_embeddings_batch(texts_to_embed, engine='text-embedding-ada-002')
movies_df['embedding'] = movies_df_embeddings


movies_df.to_csv('movieinfo-updated-embeddings pages51-54.csv')