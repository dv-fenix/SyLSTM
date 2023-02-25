export ROOT_DIR=$(pwd)
export SYLSTM_ROOT_DIR=$ROOT_DIR/sylstm
export SAVE_DIR=$ROOT_DIR/checkpoints
export DATA_DIR=$ROOT_DIR/data
export PYTHONPATH="$SYLSTM_ROOT_DIR:$PYTHONPATH"

export EPOCHS=10
export LR=1e-3
export MAX_LEN=140
export BATCH_SIZE=21
export MAX_VOCAB_SIZE=30000

# Uncomment the next line to create a new virtual_env
# python3 -m venv $VENV_DIR/sylstm_env
source $ROOT_DIR/sylstm_env/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt

cd SYLSTM_ROOT_DIR

python -u sylstm.train.py \
    -seed 4 \
    -lr $LR \
    -epochs $EPOCHS \
    -max_len $MAX_LEN \
    -save_dir $SAVE_DIR \
    -batch_size $BATCH_SIZE \
    -max_vocab_size $MAX_VOCAB_SIZE >> ./logs/training.out