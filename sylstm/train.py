import torch
import pandas as pd
import numpy as np
from transformers import AdamW, get_cosine_schedule_with_warmup
import spacy
from torchtext import data
from sylstm.eval_metrics import f1
from models.model import SyLSTM
from utils.train_utils import DataFrameDataset
from utils.parser import ArgumentParser
import utils.opts as opts
import logging
from tqdm.auto import tqdm

import os
import errno

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not os.path.isfile(args.data_dir + "/labeled_data.csv"):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            (args.data_dir + "/labeled_data.csv"),
        )

    # Set data path
    PATH = args.data_dir + "/labeled_data.csv"

    # Initialize dataframe
    df = pd.read_csv(PATH)
    df.set_index("id", inplace=True)

    logger.info("***** Initializing Dataset *****")

    # Initialize data fields
    TEXT = data.Field(
        tokenize=tokenizer,
        include_lengths=True,
        fix_length=args.max_len,
        batch_first=True,
    )
    LABEL = data.LabelField(
        dtype=torch.float, fix_length=args.max_len, batch_first=True
    )
    ADJACENCY = data.Field(fix_length=args.max_len, batch_first=True)

    fields = [("text", TEXT), ("label", LABEL), ("adjacency", ADJACENCY)]

    # split train and validation data
    train_ds, val_ds = DataFrameDataset.splits(
        fields, train_df=df[df.data_type == "train"], val_df=df[df.data_type == "valid"]
    )

    # Set device
    device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

    # Build DataIterators
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_ds, val_ds),
        batch_size=args.batch_size,
        sort_within_batch=True,
        device=device,
    )

    logger.info("***** Initializing Model *****")

    # Initialize model
    model = SyLSTM(
        vocab_size=args.max_vocab_size,
        embedding_dim=args.embedding_dim,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_hidden_layers,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        adj_weight=args.adj_wt,
        gcn_hidden_dim=args.gcn_hidden_dim,
        use_glove=args.use_glove,
    )

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-5,
        weight_decay=1e-1,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(train_iterator) * args.epochs,
    )

    if args.resume:
        logger.info("Loading pretrained model checkpoint ...")
        model.load_state_dict(torch.load(args.model_path))
    else:
        logger.info("Training new model fom scratch ...")

    # Move model to device
    model.to(device)

    logger.info("***** Running Training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size = {args.batch_size}")

    # Initialize progress bar
    progress_bar = tqdm(range(args.epochs))
    completed_steps = 0

    for epoch in range(args.epochs):
        model.train()

        train_loss = train(model, train_iterator, optimizer, scheduler)
        valid_loss, y_hat, y = evaluate(model, valid_iterator)
        weighted = f1(y_hat, y)

        logger.info(f"\tTraining Loss: {train_loss:.3f}")
        logger.info(
            f"\tValidation Loss: {valid_loss:.3f} | F1 Score (Weighted): {weighted}"
        )

        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1

        # Save Model
        MODEL_PATH = args.save_dir + f"/model_{epoch}.pth"
        torch.save(model.state_dict(), MODEL_PATH)


def train(model, iterator, optimizer, scheduler):
    """
    Runs the algorithm for a single epoch of training
    """
    epoch_loss = 0
    model.train()

    for batch in iterator:
        text, text_lengths = batch.text
        adj_vec = batch.adjacency

        optimizer.zero_grad()
        inputs = {
            "sent": text,
            "adj_vec": adj_vec,
            "target": batch.label,
            "sent_len": text_lengths,
        }

        outputs = model(**inputs)
        loss = outputs[1]
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator):
    """
    Runs the algorithm for evaluating the trained model
    """
    model.eval()
    total_eval_loss = 0
    y_hat, y = [], []

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            adj_vec = batch.adjacency

            inputs = {
                "sent": text,
                "adj_vec": adj_vec,
                "target": batch.label,
                "sent_len": text_lengths,
            }

            outputs = model(**inputs)
            loss = outputs[1]
            logits = outputs[0]
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs["target"].cpu().numpy()
            y_hat.append(logits)
            y.append(label_ids)

    y_hat = np.concatenate(y_hat, axis=0)
    y = np.concatenate(y, axis=0)

    return total_eval_loss / len(iterator), y_hat, y


def tokenizer(text: str):
    """
    Tokenize input text
    """
    return [token.text for token in nlp.tokenizer(text)]


def _get_parser():
    parser = ArgumentParser(description="train.py")
    opts.train_opts(parser)

    return parser


if __name__ == "__main__":
    main()
