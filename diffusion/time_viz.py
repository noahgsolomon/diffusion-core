import torch
import math
import matplotlib.pyplot as plt
import numpy as np

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        assert time_dim % 2 == 0, "time_dim must be even"
        
    def forward(self, t):
        device = t.device
        half_dim = self.time_dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        t = t.unsqueeze(-1).float()
        
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb

# Set up embedding function
time_dim = 64
embedding_func = SinusoidalTimeEmbedding(time_dim)

# Generate embeddings for different timesteps
timesteps = [1, 10, 50, 100, 500, 999]
embeddings = []

for t in timesteps:
    t_tensor = torch.tensor([t])
    embedding = embedding_func(t_tensor).detach().numpy()[0]
    embeddings.append(embedding)

# Plot the embeddings
plt.figure(figsize=(12, 8))
for i, (t, embedding) in enumerate(zip(timesteps, embeddings)):
    plt.subplot(len(timesteps), 1, i+1)
    plt.plot(embedding)
    plt.title(f"Embedding for timestep {t}")
    plt.ylim(-1.1, 1.1)
    
plt.tight_layout()
plt.savefig("time_embeddings.png")
plt.show()

# Visualize relationships between timesteps
plt.figure(figsize=(10, 8))
embeddings_array = np.array(embeddings)

# Calculate cosine similarity between embeddings
similarity = np.zeros((len(timesteps), len(timesteps)))
for i in range(len(timesteps)):
    for j in range(len(timesteps)):
        dot_product = np.dot(embeddings_array[i], embeddings_array[j])
        norm_i = np.linalg.norm(embeddings_array[i])
        norm_j = np.linalg.norm(embeddings_array[j])
        similarity[i, j] = dot_product / (norm_i * norm_j)

plt.imshow(similarity, cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.xticks(np.arange(len(timesteps)), timesteps)
plt.yticks(np.arange(len(timesteps)), timesteps)
plt.title("Similarity between timestep embeddings")
plt.savefig("embedding_similarity.png")
plt.show()