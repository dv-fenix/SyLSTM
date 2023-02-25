export ROOT_DIR=$(pwd)
export SYLSTM_ROOT_DIR=$ROOT_DIR/sylstm
export SAVE_PATH=$ROOT_DIR/data/labeled_data.csv
export DATA_PATH=$ROOT_DIR/data/training_data.csv
export VENV_DIR=$ROOT_DIR/venv
export PYTHONPATH="$SYLSTM_ROOT_DIR:$PYTHONPATH"


python3 -m venv $VENV_DIR/sylstm_env
source $VENV_DIR/sylstm_env/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt

cd SYLSTM_ROOT_DIR

python -u sylstm.preprocess.py \
    -max_len 140 \
    -save_path $SAVE_PATH \
    -data_path $DATA_PATH >> ./logs/preprocessing.out