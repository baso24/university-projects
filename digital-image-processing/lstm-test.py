import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Meccanismo di Self-Attention per pesare l'importanza di ogni frame (time step)
    nella sequenza di 30 celle.
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        # Un piccolo layer fully connected per calcolare i punteggi di attenzione
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs shape: (Batch, Seq_Len, Hidden_Dim)
        
        # 1. Calcola l'energia (importanza) per ogni step temporale
        energy = self.projection(encoder_outputs) # Shape: (Batch, Seq_Len, 1)
        
        # 2. Calcola i pesi (alpha) normalizzati tra 0 e 1
        weights = F.softmax(energy, dim=1) # Shape: (Batch, Seq_Len, 1)
        
        # 3. Somma pesata degli output LSTM (Context Vector)
        # Moltiplichiamo ogni output per il suo peso e sommiamo lungo la dimensione temporale
        context = torch.sum(encoder_outputs * weights, dim=1) # Shape: (Batch, Hidden_Dim)
        
        return context, weights

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(FallDetectionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. LSTM Layer
        # batch_first=True significa che l'input atteso è (Batch, Seq, Features)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=False # Puoi metterlo a True per vedere "futuro e passato"
        )
        
        # 2. Attention Block
        self.attention = SelfAttention(hidden_dim)
        
        # 3. Binary Classifier Head
        # Prende il vettore di contesto e decide: Caduta (1) o No Caduta (0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Per evitare overfitting
            nn.Linear(64, 1) # Output singolo (logit)
        )

    def forward(self, x):
        # x shape: (Batch, 30, input_dim)
        
        # Passaggio nella LSTM
        # lstm_out shape: (Batch, 30, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Applicazione dell'Attenzione
        # context_vector: riassunto pesato di tutta la sequenza
        # attn_weights: utile per visualizzare quale frame ha fatto scattare l'allarme
        context_vector, attn_weights = self.attention(lstm_out)
        
        # Classificazione
        logits = self.classifier(context_vector)
        
        return logits, attn_weights
    
def _init_():
    # Test rapido del modello
    batch_size = 2
    seq_len = 30
    input_dim = 34  # Ad esempio, 17 keypoints * 2 (x,y)
    hidden_dim = 128

    model = FallDetectionLSTM(input_dim, hidden_dim)
    sample_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(model)
    print(sample_input)

if __name__ == "__main__":
    _init_()