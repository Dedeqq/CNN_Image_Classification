import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvolutionalNetwork(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(73984, 1024)
        self.fc2 = nn.Linear(1024, 6)

        self.v_loss = []
        self.t_loss = []

    def forward(self, x):
        x = self.relu((self.conv1(x)))
        x = self.relu((self.conv2(x)))
        x = self.pool(x)

        x = self.relu((self.conv3(x)))
        x = self.relu((self.conv4(x)))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)

        return output

    def share_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.share_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.share_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     labels_hat = torch.argmax(y_hat, dim=1)
    #     n_correct_pred = torch.sum(y == labels_hat).item()
    #     loss = F.cross_entropy(y_hat, y.long())
    #     tensorboard_logs = {'train_acc_step': n_correct_pred, 'train_loss_step': loss}
    #     return {'loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(y), 'log': tensorboard_logs}
    #
    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     train_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
    #     tensorboard_logs = {'train_acc': train_acc, 'train_loss': avg_loss, 'step': self.current_epoch}
    #     self.train_losses.append(avg_loss.detach().cpu().item())
    #     self.train_accuracies.append(train_acc)
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y.long())
    #     labels_hat = torch.argmax(y_hat, dim=1)
    #     n_correct_pred = torch.sum(y == labels_hat).item()
    #     return {'val_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
    #
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
    #     tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc, 'step': self.current_epoch}
    #     self.valid_losses.append(avg_loss.detach().cpu().item())
    #     self.valid_accuracies.append(val_acc)
    #     return {'log': tensorboard_logs}

    def predict(self, data):
        y_pred = self(data)
        return torch.argmax(y_pred, dim=1)

    def training_epoch_end(self, training_step_outputs):  # list of dicts of mean loss for every batch
        epoch_av_loss = 0
        for loss in training_step_outputs:
            epoch_av_loss += loss['loss']
        self.t_loss.append(epoch_av_loss/len(training_step_outputs))

    def validation_epoch_end(self, validation_step_outputs):
        self.v_loss.append(torch.mean(torch.tensor(validation_step_outputs)))
