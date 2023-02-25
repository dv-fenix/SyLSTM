from __future__ import print_function

import configargparse


def train_opts(parser):
    """
    These options are passed to the training of the model.
    Be careful with these as they will be used during unmixing.
    """
    group = parser.add_argument_group("General")
    group.add(
        "--seed",
        "-seed",
        type=int,
        default=None,
        help="Set seed value to fix model initialization.",
    )

    group.add(
        "--resume",
        "-resume",
        action="store_true",
        help="If passed, will resume training from a checkpoint.",
    )

    group.add(
        "--model_path",
        "-model_path",
        type=str,
        default=None,
        help="Path to checkpoint if --resume is passed.",
    )

    group.add(
        "--lr",
        "-lr",
        type=float,
        default=1e-3,
        help="Learning rate for training the translator.",
    )

    group.add(
        "--num_warmup_steps",
        "-num_warmup_steps",
        type=int,
        default=0,
        help="Number of warm up steps for learning rate scheduler.",
    )

    group.add(
        "--epochs",
        "-epochs",
        type=int,
        default=10,
        help="Total number of epochs to train the model for.",
    )

    group.add(
        "--save_dir",
        "-save_dir",
        type=str,
        default="../logs",
        help="System path to save model weights.",
    )

    group.add(
        "--max_len",
        "-max_len",
        type=int,
        default=64,
        help="Maximum length of the text sequence to be processed by the model.",
    )

    group.add(
        "--embedding_dim",
        "-embedding_dim",
        type=int,
        default=300,
        help="Dimensionality of the word embedding layer of the model.",
    )

    group.add(
        "--hidden_dim",
        "-hidden_dim",
        type=int,
        default=32,
        help="Dimensionality of the hidden layer of the LSTM.",
    )

    group.add(
        "--num_hidden_layers",
        "-num_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the LSTM semantic encoder.",
    )

    group.add(
        "--max_vocab_size",
        "-max_vocab_size",
        type=int,
        default=30000,
        help="Size of the vocabulary of the trained model.",
    )

    group.add(
        "--batch_size",
        "-batch_size",
        type=int,
        default=21,
        help="Training batch size of the model.",
    )

    group.add(
        "--adj_wt",
        "adj_wt",
        type=int,
        default=5,
        help="Default edge weight denoting a dependency in the adjacency matrix.",
    )

    group.add(
        "--data_dir",
        "-data_dir",
        type=str,
        default=None,
        help="Directory containing the data for the model to be trained on.",
    )

    group.add(
        "--gcn_hidden_dim",
        "-gcn_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension for the Graph Convolutional Network.",
    )

    group.add(
        "--use_glove",
        "-use_glove",
        action="store_true",
        help="If passed, the model will be initialized using the pretrained Glove Twitter Embeddings.",
    )
