import os
import glob
import random
import shutil
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.nn.modules.loss import BCEWithLogitsLoss
import matplotlib.pyplot as plt


class BinaryClassifier:

    def __init__(self, ratios, train_size=0.85, dataset_length=200, number_of_epochs=10, data_dir='images', model_data_dir='model_data') -> None:
        self.ratios = ratios
        self.dataset_length = dataset_length
        self.n_epochs = number_of_epochs
        self.data_dir = [os.path.join(os.getcwd(), data_dir, i)
                         for i in self.ratios]
        self.model_data_dir = os.path.join(os.getcwd(), model_data_dir)
        self.train_size = train_size
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.tforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ])

        self.model, self.loss_fn, self.optimizer = self._initialize_model()

    def _initialize_model(self):
        model = models.resnet18(weights=True)

        for params in model.parameters():
            params.requires_grad = False

        # add a new final layer
        nr_filters = model.fc.in_features  # number of input features of last layer
        model.fc = nn.Linear(nr_filters, 1)

        model = model.to(self.device)

        # loss and optimizer
        # binary cross entropy with sigmoid, so no need to use sigmoid in the model
        loss_fn = BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.fc.parameters())

        return model, loss_fn, optimizer

    def create_dataset(self):
        # #remove data directory if it exists
        # if os.path.exists(self.model_data_dir):
        #     shutil.rmtree(self.model_data_dir)

        for dir in self.data_dir:
            train_dir = os.path.join(
                self.model_data_dir, 'train', dir.split('\\')[-1].split('/')[-1])
            test_dir = os.path.join(
                self.model_data_dir, 'test', dir.split('\\')[-1].split('/')[-1])

            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            imgs = sorted(glob.glob(dir + '/*'),
                          key=lambda t: os.stat(t).st_mtime)
            train_idx = int(self.dataset_length * self.train_size)
            for i, img in enumerate(imgs[:self.dataset_length]):
                if i < train_idx:
                    shutil.copy(img, train_dir)
                else:
                    shutil.copy(img, test_dir)

    def _create_dataloader(self):
        train_dir = os.path.join(self.model_data_dir, 'train')
        test_dir = os.path.join(self.model_data_dir, 'test')

        train_data = datasets.ImageFolder(train_dir, transform=self.tforms)
        self.test_data = datasets.ImageFolder(test_dir, transform=self.tforms)

        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16)  # type: ignore
        testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16)  # type: ignore

        return trainloader, testloader

    def train_step(self, trainloader):

        epoch_loss = 0
        self.model.train()
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            x_batch, y_batch = data
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(self.device)

            loss = self.loss_fn(self.model(x_batch), y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss / len(trainloader)

        return epoch_loss

    def test_step(self, testloader):
        cum_loss = 0
        correct = 0
        total = 0
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(self.device)

            self.model.eval()

            yhat = self.model(x_batch)
            val_loss = self.loss_fn(yhat, y_batch)
            cum_loss += val_loss / len(testloader)

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(yhat))
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total
        return cum_loss, accuracy

    def run(self, best_model_path='best_model.pth'):
        trainloader, testloader = self._create_dataloader()

        best_model_wts = None
        best_loss = float('inf')
        best_accuracy = 0

        accuracies = []
        epoch_train_losses = []
        epoch_test_losses = []

        early_stopping_counter = 0
        early_stopping_tolerance = 3
        early_stopping_threshold = 0.03

        for epoch in range(self.n_epochs):
            epoch_loss = self.train_step(trainloader)
            epoch_train_losses.append(epoch_loss)
            print('\nEpoch: {}, train loss: {}'.format(epoch+1, epoch_loss))

            with torch.no_grad():
                cum_loss, accuracy = self.test_step(testloader)
                epoch_test_losses.append(cum_loss)
                accuracies.append(accuracy)
                print('Epoch: {}, val loss: {}, accuracy: {:.2f}%'.format(
                    epoch+1, cum_loss, accuracy*100))

            if cum_loss <= best_loss:
                best_loss = cum_loss
                best_accuracy = accuracy
                best_model_wts = self.model.state_dict()
                torch.save(best_model_wts, best_model_path)
                print(
                    f'Best model saved at {best_model_path} with loss: {best_loss:.4f} and accuracy: {best_accuracy*100:.2f}%')

            if cum_loss > best_loss:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("\nTerminating: early stopping")
                break

    def load_best_model(self, model_path='best_model.pth'):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        return self.model

    def predict(self, model, pil_image):
        yhat = self.probability(model, pil_image)

        return round(yhat)

    def probability(self, model, pil_image):
        image = self.tforms(pil_image)
        image = image.unsqueeze(0).to(self.device)  # type: ignore
        self.model.eval()
        yhat = torch.sigmoid(self.model(image))

        return yhat.item()
    
    def inference(self, model):
        idx = torch.randint(1, len(self.test_data), (1,))
        sample = torch.unsqueeze(self.test_data[idx][0], dim=0).to(self.device)

        if torch.sigmoid(model(sample)) < 0.5:
            print("Prediction : Bottle")
        else:
            print("Prediction : Not A Bottle")


        plt.imshow(self.test_data[idx][0].permute(1, 2, 0))


if __name__ == '__main__':
    clf = BinaryClassifier('water bottle')
    model = clf.load_best_model('best_models\\water bottle.pth')
    clf.create_dataset()
    # clf.run()
    clf.inference(model)