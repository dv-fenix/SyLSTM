import torch
from torch import Tensor
import torch.nn as nn
from models.layers import GraphConvolution
from torchtext.vocab import GloVe
from utils.train_utils import get_adjacency_matrix


class SyLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_len: int,
        adj_weight: float,
        use_glove: bool,
        hidden_dim: int,
        num_layers: int,
        batch_size: int,
        num_classes: int,
        gcn_hidden_dim: int,
    ):
        """Syntax-based LSTM
        Args:
            :param vocab_size:    size of model's vocabulary
            :param embedding_dim:   dimensionality of embedding layer
            :param max_len:  maximum length of text sequence to be processed
            :param adj_weight:   edge weights of the graph
            :param hidden_dim:  dimensionality of the hidden layers of the LSTM
            :param num_layers:  number of hidden layers in the LSTM
            :param batch_size: input batch size
            :param num_classes: number of ground truth class labels
            :param gcn_hidden_dim:  dimensionality of GCN output
        Methods:
            public:
                forward(sent, sent_len, target, adj_vec) ---> classification outputs, training loss
                    Forward pass over the computational graph
        """

        # Constructor
        super(SyLSTM, self).__init__()

        self.bs = batch_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.adj_wt = adj_weight
        self.drop = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(max_len, momentum=0.6)
        self.batch2 = nn.BatchNorm1d(3 * hidden_dim, momentum=0.6)

        if use_glove:
            embedding_glove = GloVe(name="twitter.27B", dim=200)
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=200, padding_idx=0
            ).from_pretrained(embeddings=embedding_glove)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
            )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # dim of word vector
            hidden_size=hidden_dim,  # dim of output of lstm nn
            num_layers=num_layers,  # num of hidden layers
            batch_first=True,
            dropout=0.4,
            bidirectional=True,
        )

        self.gcn = GraphConvolution(2 * hidden_dim, gcn_hidden_dim)
        self.activation = nn.ReLU()
        self.linear_layer = nn.Linear((max_len * gcn_hidden_dim), hidden_dim)
        self.final_layer = nn.Linear((3 * hidden_dim), num_classes)
        self.softmax = nn.Softmax()

        # Initialize weights with Xavier Normal
        init.xavier_normal_(self.lstm.all_weights[0][0], gain=1)
        init.xavier_normal_(self.lstm.all_weights[0][1], gain=1)
        init.xavier_normal_(self.lstm.all_weights[1][0], gain=1)
        init.xavier_normal_(self.lstm.all_weights[1][1], gain=1)

        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc1.weight)

    def forward(self, sent: Tensor, sent_len: int, target: Tensor, adj_vec: Tensor):
        text_embed = self.embedding(sent)

        with torch.no_grad():
            adjMatrix = get_adjacency_matrix(
                adj_vec, self.bs, adj_vec.size()[1], self.max_len, self.adj_wt
            ).cpu()

        # Semantic Encoder
        semantic_out, (h_s, c_s) = self.lstm(text_embed, None)
        semantic_out = self.batchnorm(semantic_out)

        # Syntactic Encoder
        syntactic_out = self.gcn(semantic_out, adjMatrix, self.max_len)
        syntactic_out = self.batchnorm(syntactic_out)
        syntactic_out = self.activation(syntactic_out)
        syntactic_out = self.drop(syntactic_out)

        # Reshape
        syntactic_out = syntactic_out.view(self.bs, -1)

        syntactic_out = self.linear_layer(syntactic_out)
        syntactic_out = self.activation(syntactic_out)

        # Semantic + Syntactic (Fusion)
        final_hidden = torch.cat((h_s[-2, :, :], h_s[-1, :, :]), dim=1)
        final_hidden = final_hidden.view(self.bs, -1)

        fusion = torch.cat((final_hidden, syntactic_out), dim=1)
        fusion = self.batch2(fusion)

        output = self.final_layer(fusion)
        output = self.softmax(output)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target.long())
        return output, loss
