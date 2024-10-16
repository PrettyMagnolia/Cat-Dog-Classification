# DNN
python train.py --model_type dnn --data_dir 'data/total' --lr 0.001 --num_layers 3 --batch_size 128
python train.py --model_type dnn --data_dir 'data/total' --lr 0.001 --num_layers 4 --batch_size 128
python train.py --model_type dnn --data_dir 'data/total' --lr 0.001 --num_layers 5 --batch_size 128

# CNN
python train.py --model_type cnn --data_dir 'data/total' --lr 0.001 --num_layers 3 --batch_size 128
python train.py --model_type cnn --data_dir 'data/total' --lr 0.001 --num_layers 4 --batch_size 128
python train.py --model_type cnn --data_dir 'data/total' --lr 0.001 --num_layers 5 --batch_size 128

# RNN
python train.py --model_type rnn --data_dir 'data/total' --lr 0.001 --num_layers 3 --batch_size 128
python train.py --model_type rnn --data_dir 'data/total' --lr 0.001 --num_layers 4 --batch_size 128
python train.py --model_type rnn --data_dir 'data/total' --lr 0.001 --num_layers 5 --batch_size 128

