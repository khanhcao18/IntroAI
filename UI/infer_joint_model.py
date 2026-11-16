import torch
import torch.nn as nn
from torch.nn.functional import softmax


aspect2idx = {
    'CAMERA': 0, 'FEATURES': 1, 'BATTERY': 2, 'PERFORMANCE': 3,
    'DESIGN': 4, 'GENERAL': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 2, 'Neutral': 1, 'Negative': 0
}
idx2aspect = {index: label for label, index in aspect2idx.items()}
idx2sentiment = {index: label for label, index in sentiment2idx.items()}


class HierachyAttention(nn.Module):
    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)  # (batch_size, seq_len, out_features)
        u = torch.tanh(u)
        similarities = self.uw(u)  # (batch_size, seq_len, 1)
        similarities = similarities.squeeze(dim=-1)  # (batch_size, seq_len)

        # Mask the similarities
        similarities = similarities.masked_fill(~mask.bool(), -float('inf'))

        if self.softmax:
            alpha = torch.softmax(similarities, dim=-1)
            return alpha
        else:
            return similarities

def element_mul(input1, input2, return_not_sum_result=False):
        output = input1 * input2.unsqueeze(2)  # Ensure correct broadcasting
        result = output.sum(dim=1)
        if return_not_sum_result:
            return result, output
        else:
            return result


class JointModel(nn.Module):
    def __init__(self, word_embedder, categories, polarities):
        super().__init__()
        self.word_embedder = word_embedder
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

        embed_dim = word_embedder.embedding_dim
        self.embedding_layer_fc = nn.Linear(embed_dim, embed_dim)
        self.embedding_layer_aspect_attentions = nn.ModuleList([HierachyAttention(embed_dim, embed_dim) for _ in range(self.category_num)])
        self.lstm_layer_aspect_attentions = nn.ModuleList([HierachyAttention(embed_dim, embed_dim) for _ in range(self.category_num)])

        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(self.category_num)])
        self.sentiment_fc = nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, self.polarity_num))

    def forward(self, tokens, labels, mask, threshold=0.4):
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)

            category_output = element_mul(embeddings, alpha)
            embedding_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), word_embeddings).squeeze(1)
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)

        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            category_output = element_mul(lstm_result, alpha)
            lstm_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), lstm_result).squeeze(1)
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]], dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)
        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.category_num]
            polarity_labels = labels[:, self.category_num:]

            for i in range(self.category_num):
                category_mask = (category_labels[:, i] != -1)  # Mask out ignored labels
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  # Only calculate if there are valid labels
                    category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(-1)[category_mask], category_labels[:, i][category_mask])
                    loss += category_temp_loss

                if sentiment_mask.any():  # Only calculate if there are valid labels
                    sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i][sentiment_mask], polarity_labels[:, i][sentiment_mask].long())
                    loss += sentiment_temp_loss


        # formatting output
        final_category_outputs = [torch.sigmoid(e) for e in final_category_outputs]
        final_sentiment_outputs = [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        final_sentiment_outputs = [torch.argmax(e, dim=-1) for e in final_sentiment_outputs]
        
        final_categories = []
        final_sentiments = []

        for i in range(len(final_category_outputs)):
            batch_category = []
            batch_sentiment = []
            for j, category_score in enumerate(final_category_outputs[i]):
                # Apply threshold for aspect detection
                if category_score >= threshold:
                    batch_category.append(1)  # Aspect detected
                    batch_sentiment.append(final_sentiment_outputs[i][j].item())
                else:
                    batch_category.append(0)  # Aspect not detected
                    batch_sentiment.append(-1)  # Set sentiment to -1 for undetected aspect
            final_categories.append(batch_category)
            final_sentiments.append(batch_sentiment)
        final_categories = torch.tensor(final_categories)
        final_sentiments = torch.tensor(final_sentiments)
        
        output = {
            'pred_category': torch.transpose(final_categories, 0, 1), # batch_size*10
            'pred_sentiment': torch.transpose(final_sentiments, 0, 1) # batch_size*10
        }

        return output
