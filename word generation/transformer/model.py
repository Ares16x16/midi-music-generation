import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, embedding_dim, seq_length):
        super(SelfAttentionHead, self).__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(seq_length, seq_length)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        scores = q @ k.transpose(-2, -1) * C**-0.5
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, seq_length):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size, embedding_dim, seq_length)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(num_heads * head_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(FeedForwardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.network(x)

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, seq_length, embedding_dim):
        super(TransformerBlock, self).__init__()
        head_size = embedding_dim // num_heads
        self.attention = MultiHeadSelfAttention(num_heads, head_size, embedding_dim, seq_length)
        self.feed_forward = FeedForwardNetwork(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=100, embedding_dim=32, seq_length=8, num_heads=4, num_layers=4):
        super(Transformer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(seq_length, embedding_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(num_heads, seq_length, embedding_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_emb = self.token_embeddings(idx)
        position_emb = self.position_embeddings(torch.arange(T, device=idx.device))
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.token_embeddings.num_embeddings:]
            logits, _ = self.forward(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx