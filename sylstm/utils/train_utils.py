import torch
from torch import Tensor
import spacy
from torchtext import data
import numpy as np


# pad to max_len
def pad_list(adj_vec: list, pad_len: int):
    """
    Function to pad/truncate the input sequence into the specified length
    """
    # pop elements to conform to max_len
    while len(adj_vec) > pad_len:
        adj_vec.pop()
    while len(adj_vec) < pad_len:
        # Elements labelled -1 will be ignored in the adjacency matrix construct
        # Signals no dependence
        adj_vec.append(-1)
    return adj_vec


def get_adjacency_vector(tweet: str, pad_len: int):
    """
    Function to extract the Dependency Parse Tree of a given tweet.
    Converts the DPT into a vector form which can then be converted into a graph on the fly.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(tweet)
    vec = []
    word_dict = {}
    for i, token in enumerate(doc):
        # initialize vector
        vec.append(-1)
        # words with position indexes
        word_dict[token] = i

    # Children store positional indexes of parent nodes
    for i, token in enumerate(doc):
        if len([child for child in token.children]) == 0:
            continue
        for child in token.children:
            vec[word_dict[child]] = i

    # pad to max_len
    adj_vec = pad_list(vec, pad_len)
    return adj_vec


def get_adjacency_matrix(
    adj_vec: Tensor, batch_size: int, vec_dim: int, max_len: int, adj_weight: float
):
    """
    Funcion that converts the DPT in vector form into the matric representation of the graph.
    """

    # Check input validity
    assert (
        adj_vec.size()[0] == batch_size
    ), "Batch Size and Input Vector dimension mismatch"
    assert (
        adj_vec.size()[1] == vec_dim
    ), "Input vector dimensional mismatch, check vec_dim or input vector shape"

    # initialization
    adj_mat = torch.zeros((batch_size, vec_dim, vec_dim))

    # identity matrix for self dependency
    identity = (
        torch.matrix_power(adj_mat[0, :, :], 0) * adj_weight
    )  # dim: (vec_dim, vec_dim)

    # batch adjacency matrix
    for i in range(batch_size):
        for j in range(vec_dim):
            idx = adj_vec[i, j]
            # for roots in dependency tree and padded arguments
            if idx == -1:
                continue
            # set integer to max_len-1
            if idx > (max_len - 1):
                continue
            adj_mat[i, j, idx] = adj_weight  # indicates dependence
            adj_mat[i, idx, j] = adj_weight  # symmetric dependence

    return (
        adj_mat + identity
    )  # Broadcasting: (batch_size, vec_dim, vec_dim) + (vec_dim, vec_dim)


class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        """Dataset Class
        Args:
            :param df:    input pandas dataframe
            :param fields:   expected data fields
            :param is_test:  inference phase?
        Methods:
            public:
                sort_key(ex) ---> length of sequence
                     Key used to sort the dataset
                splits(fields, train_df, val_df) ---> (training_data, validation_data)
                    Generates samples for iterator
        """
        examples = []
        for i, row in df.iterrows():
            label = row.labels if not is_test else None
            text = row.tweet
            adjacency = row.adjacency
            examples.append(data.Example.fromlist([text, label, adjacency], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, **kwargs):
        train_data, val_data = (None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)

        return tuple(d for d in (train_data, val_data) if d is not None)
