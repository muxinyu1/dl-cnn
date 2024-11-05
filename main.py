import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import medmnist
from medmnist import INFO, Evaluator
from ranger_adabelief import RangerAdaBelief
import wandb
import os
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms

def run(model, train_set, valid_set, test_set, criterion, optimizer, scheduler, save_dir, data_path, num_epochs=20, patience=5):
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        bar = tqdm(train_loader, desc='train')
        for inputs, labels in bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.squeeze().long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            bar.set_postfix(loss=loss.item())
        epoch_loss = total_loss / len(train_loader.dataset)
        return epoch_loss

    def valid_or_test(model, valid_loader, split):
        model.train(False)
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        for inputs, labels in tqdm(valid_loader, desc=split):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            labels = labels.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            labels = labels.float().resize_(len(labels), 1)
            y_true = torch.cat((y_true, labels.cpu()), 0)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
        scheduler.step()
        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, split, size=64, root=data_path)
        metrics = evaluator.evaluate(y_score)
        return metrics[0], metrics[1]

    best_acc = 0.0
    epochs_without_improvement = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb.init(project="medmnist-training", config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "model": "swin_transformer",
        "data": data_flag
    })

    for epoch in range(num_epochs):
        print(f'epoch: {epoch+1}/{num_epochs}')
        print('*' * 100)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
        train_loss = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}".format(train_loss))

        # Log training loss to wandb
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=10)
        with torch.no_grad():
            val_auc, val_acc = valid_or_test(model, valid_loader, 'val')
        print(f'valid auc: {val_auc:.3f}  acc: {val_acc:.3f}')

        # Log validation metrics to wandb
        wandb.log({"val_auc": val_auc, "val_acc": val_acc, "epoch": epoch})

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=10)
    with torch.no_grad():
        test_auc, test_acc = valid_or_test(model, test_loader, 'test')
    print(f'test auc: {test_auc:.3f}  acc: {test_acc:.3f}')

    # Log test metrics to wandb
    wandb.log({"test_auc": test_auc, "test_acc": test_acc})

    # Save the best model
    torch.save(best_model, os.path.join(save_dir, 'best_model.pt'))
    wandb.save(os.path.join(save_dir, 'best_model.pt'))
    
    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hw1')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
    args = parser.parse_args()

    data_flag = 'pathmnist'
    download = False
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    data_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    data_transform_valid_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    data_path = '../'
    train_dataset = DataClass(root=data_path, split='train', transform=data_transform_train, size=64, download=download)
    valid_dataset = DataClass(root=data_path, split='val', transform=data_transform_valid_test, size=64, download=download)
    test_dataset = DataClass(root=data_path, split='test', transform=data_transform_valid_test, size=64, download=download)

    num_epochs = 30
    lr = 0.001
    batch_size = 32

    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = RangerAdaBelief(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    run(model, train_dataset, valid_dataset, test_dataset, criterion, optimizer, scheduler, args.save_dir, data_path, num_epochs=num_epochs)
