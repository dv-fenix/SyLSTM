from __future__ import print_function

import configargparse


def preprocess_opts(parser):
    """
    These options are passed to the training of the model.
    Be careful with these as they will be used during unmixing.
    """
    group = parser.add_argument_group("General")

    group.add(
        "--data_path",
        "-data_path",
        type=str,
        default=None,
        help="Path to the dataset to be preprocessed.",
    )

    group.add(
        "--save_path",
        "-save_path",
        type=str,
        default=None,
        help="Path to the save the preprocessed data.",
    )

    group.add(
        "--max_len",
        "-max_len",
        type=int,
        default=64,
        help="Maximum length of the text sequence to be processed by the model.",
    )
