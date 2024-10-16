import argparse
import os

import torch
from models import DNN, CNN, RNN
from dataloader import load_data


def infer(model, data_loader, args):
    model.eval()  # 设置模型为评估模式
    predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference using a pre-trained model for cat and dog classification.')
    parser.add_argument('--model_type', type=str, choices=['dnn', 'cnn', 'rnn'],
                        help='Model type to use: dnn, cnn, or rnn.')
    parser.add_argument('--data_dir', type=str, help='Directory of the dataset.')
    parser.add_argument('--image_size', type=int, default=200, help='Size of the image after transformation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference.')
    parser.add_argument('--load_path', type=str, help='Path to load the model weights.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes(cat and dog).')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the neural network.')
    args = parser.parse_args()

    # 检查模型路径是否存在
    if args.load_path is None or not os.path.exists(args.load_path):
        raise FileNotFoundError(f'Model weights file {args.load_path} does not exist.')

    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    _, test_loader = load_data(args.data_dir, args.batch_size, args.image_size)

    # 动态创建指定模型的类(eg. model_type: cnn -> model: CNN())
    model = eval(args.model_type.upper())(args=args).to(args.device)

    # 加载模型权重
    model.load_state_dict(torch.load(args.load_path, map_location=args.device))

    # 执行推理
    predictions = infer(model=model, data_loader=test_loader, args=args)

    # 输出预测结果
    print('Predictions:', predictions)
