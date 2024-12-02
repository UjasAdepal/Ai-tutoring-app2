import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import torch
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from transformers.utils.import_utils import is_torch_tpu_available
# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
# openai_api_key = os.environ.get("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")

# Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Load the dataset
data = pd.read_csv("temp2.csv", encoding='latin1')

# Extract questions and answers
questions = data['Q'].tolist()
answers = data['A'].tolist()

# Generate embeddings for the questions
print("Generating embeddings...")
question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)

# Semantic search function
def semantic_search(query, embeddings, questions, answers, top_k=3):
    # Generate the embedding for the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    
    # Get the top_k results
    top_results = torch.topk(similarities, k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((questions[idx], answers[idx], float(score)))
    return results

# Setup the LLMChain and prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key="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")


template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practices, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practices, 
in terms of length, tone of voice, logical arguments, and other details.

2/ If the best practices are irrelevant, then try to mimic the style of the best practice to prospect's message.

Below is a message I received from the prospect:
{message}

Here is a list of best practices of how we normally respond to prospects in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Retrieval augmented generation
def generate_response(message):
    search_results = semantic_search(message, question_embeddings, questions, answers, top_k=3)
    best_practices = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in search_results])
    response = chain.run(message=message, best_practice=best_practices)
    return response

# Streamlit app
def main():
    st.set_page_config(page_title="Customer Response Generator", page_icon=":bird:")

    st.header("Customer Response Generator :bird:")
    message = st.text_area("Enter the customer message:")

    if message:
        st.write("Generating response...")
        response = generate_response(message)
        st.info(response)

if __name__ == '__main__':
    main()
