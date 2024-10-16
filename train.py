# utils/train.py
import argparse
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import *
from dataloader import load_data


def train(model, train_loader, val_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(args.epochs):
        model.train()
        correct, running_loss = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        # summary train metric
        train_loss = running_loss / len(train_loader)
        writer.add_scalar('train/Loss', train_loss, epoch + 1)
        print(f'Train: Epoch [{epoch + 1}/{args.epochs}], Loss: {train_loss:.4f}')

        # evaluate
        model.eval()  # 设置模型为评估模式
        correct, running_loss = 0, 0
        correct_cat, correct_dog, total_cat, total_dog = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # 计算猫和狗的准确率
                for i in range(len(labels)):
                    if labels[i] == 0:  # 猫类
                        if predicted[i] == labels[i]:
                            correct_cat += 1
                        total_cat += 1
                    elif labels[i] == 1:  # 狗类
                        if predicted[i] == labels[i]:
                            correct_dog += 1
                        total_dog += 1

        # summary evaluate metric
        val_loss = running_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        # 计算猫和狗的准确率
        cat_acc = correct_cat / total_cat if total_cat > 0 else 0
        dog_acc = correct_dog / total_dog if total_dog > 0 else 0

        # 记录到TensorBoard
        writer.add_scalar('val/Loss', val_loss, epoch + 1)
        writer.add_scalar('val/Acc', val_acc, epoch + 1)
        writer.add_scalar('val/Cat_Acc', cat_acc, epoch + 1)
        writer.add_scalar('val/Dog_Acc', dog_acc, epoch + 1)

        # 输出结果
        print(f'Val: Epoch [{epoch + 1}/{args.epochs}], Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Cat Acc: {cat_acc:.4f}, Dog Acc: {dog_acc:.4f}')
    print("Train Finish!")
    if args.save_model:
        torch.save(model.state_dict(), args.save_path)
        print(f'Model weights saved to {args.save_path}')

    # 保存结果
    with open('./result.txt', 'a') as f:
        f.write(f'{args.model_type} {args.data_dir} {args.lr} {args.num_layers} {val_acc} {cat_acc} {dog_acc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for cat and dog classification.')
    parser.add_argument('--model_type', type=str, choices=['dnn', 'cnn', 'rnn'],
                        help='Model type to use: dnn, cnn, or rnn.')
    parser.add_argument('--data_dir', type=str, help='Directory of the dataset.')
    parser.add_argument('--image_size', type=int, default=200, help='Size of the image after transformation.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--log_dir', type=str, help='Directory for TensorBoard logs.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes(cat and dog).')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the neural network.')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='Whether to save the model weights after training.')
    parser.add_argument('--save_path', type=str, help='Path to save the model weights.')
    args = parser.parse_args()

    # args.model_type = 'dnn'
    # args.data_dir = './data/total'
    # 设置 tensorboard 缓存目录
    if args.log_dir is None:
        dateset_name = args.data_dir.split('/')[-1]
        args.log_dir = f'./logs/{args.model_type}/dataset_{dateset_name}_lr_{args.lr}_num_layers_{args.num_layers}'
    # 设置模型保存路径
    if args.save_model and args.save_path is None:
        dateset_name = args.data_dir.split('/')[-1]
        args.save_path = f'./checkpoints/{args.model_type}/dataset_{dateset_name}_lr_{args.lr}_num_layers_{args.num_layers}/model.pth'
        if not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader = load_data(args.data_dir, args.batch_size, args.image_size)

    # 动态创建指定模型的类(eg. model_type: cnn -> model: CNN())
    model = eval(args.model_type.upper())(args=args).to(args.device)

    # 训练模型
    train(model=model, train_loader=train_loader, val_loader=val_loader, args=args)
