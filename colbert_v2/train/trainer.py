# colbert_v2/train/trainer.py
import torch
from torch import nn, optim

class Trainer:
    def __init__(self, encoder, dataloader):
        self.encoder = encoder
        self.dataloader = dataloader
        self.loss_fn = nn.MarginRankingLoss(margin=1.0)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=Config.LEARNING_RATE)

    def train(self, epochs=Config.EPOCHS):
        for epoch in range(epochs):
            for batch in self.dataloader:
                queries, positives, negatives = batch  # Positive/negative pairs
                pos_emb = self.encoder.encode(positives)
                neg_emb = self.encoder.encode(negatives)
                query_emb = self.encoder.encode(queries)
                
                pos_sim = LateInteraction.compute_similarity(query_emb, pos_emb)
                neg_sim = LateInteraction.compute_similarity(query_emb, neg_emb)
                
                # Contrastive loss: higher similarity for positives
                loss = self.loss_fn(pos_sim, neg_sim, torch.ones_like(pos_sim))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
