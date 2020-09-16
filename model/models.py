import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Encoder Model
class Conv_Encoder(nn.Module):
    def __init__(self, kernel_size=1, embedding_dim=620, encoder_dim=2400, sentence_len=32):
        super(Conv_Encoder, self).__init__()

        conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = encoder_dim, kernel_size = 1)
        # Section 3.3: sentence encoder genc (a 1D-convolution + ReLU + mean-pooling)
        self.conv_blocks = nn.Sequential(
            conv1d,
            nn.ReLU(),
            nn.AvgPool1d(kernel_size = sentence_len)
        )

    def forward(self, x): # x is (B,S,D)      
        x = x.transpose(1,2)  # needs to convert x to (B,D,S)
        feature_extracted = self.conv_blocks(x) # feature_extracted is (B,D)
        
        return feature_extracted.squeeze()

# Contrastive Predictive Coding Model
class CPCv1(nn.Module):
    def __init__(self, config):
        super(CPCv1, self).__init__()
        # load parameters
        self.enc_dimension = config.cpc_model.enc_dim
        self.ar_dimension = config.cpc_model.ar_dim
        self.k_size = config.cpc_model.k_size
        # define embedding layer
        self.embedding = nn.Embedding(
            config.dataset.vocab_size, 
            config.cpc_model.emb_dim, 
            padding_idx=config.dataset.padding_idx
        )
        # define type of encoder
        self.encoder = Conv_Encoder(
            kernel_size=1, 
            embedding_dim=config.cpc_model.emb_dim, 
            encoder_dim=config.cpc_model.enc_dim, 
            sentence_len=config.dataset.max_sen_length
        )
        # define autoregressive model
        self.gru = nn.GRU(
            self.enc_dimension, 
            self.ar_dimension, 
            batch_first=True)
        # define predictive layer
        self.Wk  = nn.ModuleList([nn.Linear(self.enc_dimension, self.ar_dimension) for i in range(self.k_size)])

    def init_hidden(self, batch_size, device = None):
        if device: return torch.zeros(1, batch_size, self.ar_dimension).to(device)
        else: return torch.zeros(1, batch_size, self.ar_dimension)
    
    # B: Batch, W: Window, S: Sentence, D: Dimension
    def forward(self, x): # x is (B,W,S)
        batch, window, sentence_length = x.shape
        device = x.device
        # create dummy hidden state
        hidden = self.init_hidden(batch, device) # hidden is (B,D)
        # get sentence embeddings
        z = self.get_sentence_embedding(x.view(batch*window, sentence_length)) # z is (B*W,D)
        z = z.view(batch, window, self.enc_dimension) # z is (B,W,D)
        # separate forward and target samples
        # W1: forward window, W2: target window
        target = z[:,-self.k_size:,:].transpose(0,1) # target is (W2,B,D)
        forward_sequence = z[:,:-self.k_size,:] # forward_sequence is (B,W1,D)
        # feed ag model
        self.gru.flatten_parameters()
        output, hidden = self.gru(forward_sequence, hidden) # output is (B,W1,D)
        context = output[:,-1,:].view(batch, self.ar_dimension) # context is (B,D) (take last hidden state)
        pred = torch.empty((self.k_size, batch, self.enc_dimension), dtype = torch.float, device = device) # pred (empty container) is (W2,B,D)
        # loop of prediction
        for i in range(self.k_size):
            linear = self.Wk[i]
            pred[i] = linear(context) # Wk*context is (B,D)
        loss, accuracy = self.info_nce(pred, target)

        return loss, accuracy 

    def get_sentence_embedding(self, x): # x is (B,S)
        out = self.embedding(x) # out is (B,S,D)
        z = self.encoder(out) # z is (B,D)
        
        return z
    
    def get_word_embedding(self, x):
        word_embeddings = self.embedding(x)
        
        return word_embeddings
    
    def info_nce(self, prediction, target):
        k_size, batch_size, hidden_size = target.shape
        label = torch.arange(0, batch_size * k_size, dtype=torch.long, device=target.device)
        # compute nce
        logits = torch.matmul(
            prediction.reshape([-1, hidden_size]), 
            target.reshape([-1, hidden_size]).transpose(-1, -2)
        )
        loss = nn.functional.cross_entropy(logits, label, reduction='none')
        accuracy = torch.eq(
            torch.argmax(F.softmax(logits, dim = 1), dim = 1),
            label)
        # process for split loss and accuracy into k pieces (useful for logging)
        nce, acc = [], []
        for i in range(k_size):
            start = i * batch_size
            end = i * batch_size+batch_size
            nce.append(torch.sum(loss[start:end]) / batch_size)
            acc.append(torch.sum(accuracy[start:end], dtype=torch.float) / batch_size)
        
        return torch.stack(nce).unsqueeze(0), torch.stack(acc).unsqueeze(0)
    
class TxtClassifier(nn.Module):
    ''' linear classifier '''
    def __init__(self, config):
        super(TxtClassifier, self).__init__()
        self.classifier = nn.Linear(config.txt_classifier.d_input, config.txt_classifier.n_class)

    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
