import pandas as pd
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_excel('Coursera Enterprise Catalog_Certificate Filter - 2024-25.xlsx', sheet_name='All Enterprise Courses')

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define career interests
career_interests = ['deployment', 'blockchain', 'cybersecurity']
career_interest_embeddings = model.encode(career_interests)

# Encode course descriptions
df['description_embedding'] = df['Course Description'].apply(lambda x: model.encode([x])[0])

# Calculate cosine similarity between course descriptions and career interests
def calculate_relevance(description_embedding, career_interest_embeddings):
    relevance_scores = [cosine_sim(description_embedding, interest_embedding) for interest_embedding in career_interest_embeddings]
    return max(relevance_scores)

def cosine_sim(a, b):
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

df['relevance_score'] = df.apply(lambda row: calculate_relevance(row['description_embedding'], career_interest_embeddings), axis=1)

# Sort courses by relevance score
top_courses = df.nlargest(20, 'relevance_score')[['Course Name', 'relevance_score']]

print(top_courses)