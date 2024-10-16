# DNN
python train.py --model_type dnn --data_dir 'data/mini' --lr 0.001 --num_layers 3
python train.py --model_type dnn --data_dir 'data/mini' --lr 0.001 --num_layers 4
python train.py --model_type dnn --data_dir 'data/mini' --lr 0.001 --num_layers 5

# CNN
python train.py --model_type cnn --data_dir 'data/mini' --lr 0.001 --num_layers 3
python train.py --model_type cnn --data_dir 'data/mini' --lr 0.001 --num_layers 4
python train.py --model_type cnn --data_dir 'data/mini' --lr 0.001 --num_layers 5

# RNN
python train.py --model_type rnn --data_dir 'data/mini' --lr 0.001 --num_layers 3
python train.py --model_type rnn --data_dir 'data/mini' --lr 0.001 --num_layers 4
python train.py --model_type rnn --data_dir 'data/mini' --lr 0.001 --num_layers 5