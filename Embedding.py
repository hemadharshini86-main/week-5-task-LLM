from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Read the text file
with open('Text_Embedding/sentences.txt', 'r', encoding='utf-8') as file:
    texts = [line.strip() for line in file if line.strip()]

print("Number of Texts:", len(texts))

# 2. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Generate embeddings
embeddings = model.encode(texts)

print("Embeddings shape:", embeddings.shape)

# 4. Save embeddings
np.save('Text_Embedding/embeddings.npy', embeddings)

print("Embeddings saved successfully!")
# 5. Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)

print("Similarity matrix shape:", similarity_matrix.shape)

# 6. Create sentence pairs
pairs = []

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        pairs.append({
            "Text 1": texts[i],
            "Text 2": texts[j],
            "Cosine Similarity": round(similarity_matrix[i][j], 4)
        })

# 7. Convert to DataFrame
import pandas as pd

results_df = pd.DataFrame(pairs)

# 8. Save similarity results
results_df.to_csv(
    'Text_Embedding/data/similarity_results.csv',
    index=False
)

print("Similarity results saved successfully!")

# 9. Display top 5 most similar pairs
top_5 = results_df.sort_values(
    by="Cosine Similarity",
    ascending=False
).head(5)

print("\nTop 5 Most Similar Pairs:")
print(top_5.to_string(index=False))