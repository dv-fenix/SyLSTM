from nltk import TweetTokenizer
import splitter
import spacy
import emoji
from utils.parser import ArgumentParser
import utils.preprocess_opts as opts
import logging
import pandas as pd
from utils.train_utils import get_adjacency_vector
from tqdm.auto import tqdm

import os
import errno

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


def main():
    ###### Add Argparser #######
    parser = _get_parser()
    args = parser.parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), (args.data_path)
        )

    logger.info("***** Loading Dataset *****")
    logger.info(f" Loading data from: {args.data_path}")

    # Load Data
    df = pd.read_csv(args.data_path, sep="\t")
    df.set_index("id", inplace=True)

    logger.info("***** Preprocessing Dataset *****")
    # Replace Emojis in tweets
    df["tweet"] = replaceEmojis(df.tweet.values)

    # Further pre-processing and tokenization
    df["tweet"] = tokenize(df.tweet.values)

    logger.info("***** Finished Preprocessing Dataset *****")
    logger.info(f" Number of tweets pre-processed: {df.tweet.value_counts()}")

    # Initialize progress bar
    progress_bar = tqdm(range(df.tweet.value_counts()))
    completed_steps = 0

    # Intialize list to store outputs of adjacency vectors
    output_vectors = []

    logger.info("***** Extracting Dependency Parse Trees *****")
    for tweet in df.tweet.values.to_list():
        vector = get_adjacency_vector(tweet, args.max_len)
        output_vectors.append(vector)

        # Update Progress
        progress_bar.update(1)
        completed_steps += 1

    logger.info(f" Extracted {len(output_vectors)} DPTs")
    logger.info("***** Finished Extracting DPTs *****")

    # Save pre-processed data
    df["adjacency"] = output_vectors
    df.to_csv(args.save_path, sep="\t")


def replaceEmojis(sentences: list):
    """
    Function to replace the emoji's in a tweet with their text-based counterparts.
    """
    output = []
    for item in sentences:
        output.append(emoji.demojize(item))

    return output


def preprocess(word):
    """
    Function to perform the following pre-processing steps for a given tweet:
        1. Hashtag Segmentation
        2. Compound Word Splitting
        3. Reducing Word Lengths
    """
    for char in word:
        temp = []
        # Hashtag Segmentation
        if char == "#":
            word = word[1:]

            # Compund word splitter
            if splitter.split(word) != "":
                [temp.append(letter) for letter in splitter.split(word)]
                str_2 = "# "
                for letter in temp:
                    str_2 = str_2 + letter + " "
                str_2 = str_2[:-1]
                return str_2
            else:
                return "# " + word
        else:
            return word


def listToString(sentence: list):
    """
    Replaces usernames in strings and prompts the script to conduct further pre-processing.
    """
    # initialize an empty string
    str1 = ""

    count = 0
    # traverse the string
    for word in sentence:
        if word == "@USER":
            count += 1
            if count > 3:
                continue

        # Further pre-processing steps
        word = preprocess(word)
        str1 = str1 + word + " "

    # remove \s at the end of string
    str1 = str1[:-1]

    # return string
    return str1


def tokenize(sentences: list):
    """
    Function to preprocess and tokenize all the tweets in the dataset.
    """
    # Initialize tokenizer to reduce emphasis and lower case tweets
    output = []
    tokenizer = TweetTokenizer(reduce_len=True, preserve_case=False)

    for item in sentences:
        tokens = tokenizer.tokenize(item)
        sentence = listToString(tokens)
        output.append(sentence)

    return output


def _get_parser():
    """
    Private function to initialize command line argument parser.
    """
    parser = ArgumentParser(description="preprocess.py")
    opts.preprocess_opts(parser)

    return parser


if __name__ == "__main__":
    main()
