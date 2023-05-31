from datetime import datetime
from time import time
from functools import partial

import torch
import matplotlib.pyplot as plt
import numpy as np

from loss import IoUScore, DiceLoss
from model import DecoderBlock


def timestamp():
    return datetime.fromtimestamp(time()).strftime("%d-%m-%Y-%H:%M:%S")


class Model:
    def __init__(
        self,
        model,
        test_loader,
        train_loader,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
        lr=1e-3,
        weight_decay=0,
        metric=IoUScore(),
        writer=None,
        pics_to_log_demonstration=None,
        pics_to_log_activations=None,
        pics_to_log_gradients=None,
        log_interval_iter=150,
        log_interval_epoch=1,
    ):
        # UNet
        self.model = model.to(device)
        self.device = device

        # which loss to optimize
        self.criterion = criterion

        # data
        self.test_loader = test_loader
        self.train_loader = train_loader

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

        # metric for val set evaluation
        self.metric = metric

        # tensorboard writer
        self.writer = writer

        # to demonstrate work of model during training and to log activations and gradients
        self.pics_to_log_gradients = pics_to_log_gradients
        self.pics_to_log_demonstration = pics_to_log_demonstration
        self.pics_to_log_activations = pics_to_log_activations
        self.log_interval_iter = log_interval_iter
        self.log_interval_epoch = log_interval_epoch

    @torch.no_grad()
    def val(self):
        self.model.eval()

        # collect loss and metric all over the test set
        metr = []
        loss = []
        for inputs, labels in self.test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)

            loss += self.criterion(outputs, labels).cpu().tolist()
            metr += self.metric(outputs, labels).cpu().tolist()

        # compute mean
        loss = np.mean(loss)
        metr = np.mean(metr)

        return loss, metr

    @torch.no_grad()
    def log_epoch(self, epoch, loss, metr):
        if epoch % self.log_interval_epoch != 0:
            return

        self.writer.add_scalar('Loss/Test', loss, epoch)
        self.writer.add_scalar('Metric/Test', metr, epoch)

        # update best params
        if metr > self.best_metr:
            self.best_epoch = epoch
            self.best_metr = metr

        # checkpoint
        path = 'check-' + timestamp() + '.pth'
        torch.save(self.model.state_dict(), path)
        print(f'Epoch {epoch}: saved checkpoint to {path}')

        # model params distribution
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.writer.add_histogram(
                    f'Hist/Weight[{name}]', module.weight, epoch)
                self.writer.add_histogram(
                    f'Hist/Bias[{name}]', module.bias, epoch)
                if epoch != 0:
                    self.writer.add_histogram(
                        f'Weight/{name}', module.weight.grad, epoch)
                    self.writer.add_histogram(
                        f'Bias/{name}', module.bias.grad, epoch)

        # demonstration of model's work
        self.writer.add_figure('Demonstration', self.figure_to_log(), epoch)

    @torch.no_grad()
    def log_activations(self, epoch):
        # batch of images to be investigated
        input, _ = self.pics_to_log_activations

        # define hook function
        def hook(name, module, inp, out):
            for i in range(out.shape[0]):   # for each object in batch
                # visualize mean of feature maps
                self.writer.add_image(
                    f'Activation[{i}]/module[{name}]', out[i].mean(dim=0),
                    epoch, dataformats='HW'
                )

        # register hooks for encoder and decoder blocks
        handlers = []
        for name, layer in self.model.named_modules():
            is_enc_block = isinstance(
                layer, torch.nn.Sequential)   # encoder block
            is_dec_block = isinstance(layer, DecoderBlock)
            if not (is_enc_block or is_dec_block):
                continue

            layer.__name__ = name
            handler = layer.register_forward_hook(partial(hook, name))
            handlers.append(handler)

        # feed to model
        self.model.eval()
        self.model(input.to(self.device))

        # disable hooks
        for handler in handlers:
            handler.remove()

    def train(self, num_epoch):
        # initial model evaluation
        self.best_metr = 0
        self.best_epoch = 0
        test_loss, test_metr = self.val()
        self.log_epoch(0, test_loss, test_metr)

        # launch train
        for epoch in range(1, num_epoch+1):
            for i, (inputs, labels) in enumerate(self.train_loader):
                # for logging
                iter_num = i * len(inputs) + (epoch - 1) * len(self.train_loader.dataset)

                # send data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels).mean()
                metr = self.metric(outputs, labels).mean()

                self.log_iter(iter_num, loss.item(), metr.item())

                # backward pass
                loss.backward()
                self.optimizer.step()

            # evaluate at the end of an epoch
            test_loss, test_metr = self.val()
            self.log_epoch(epoch, test_loss, test_metr)
            self.log_activations(epoch)
            self.log_gradients(epoch)

        self.log_final_model(num_epoch, test_loss, test_metr)

    def log_final_model(self, num_epoch, loss, metr):
        # hyperparameters
        self.writer.add_hparams(
            hparam_dict={
                'optimizer': str(self.optimizer),
                'criterion': str(self.criterion),
                'num_epoch': num_epoch,
            },
            metric_dict={
                'final_loss': loss,
                'final_metric': metr,
                'best_metr': self.best_metr,
                'best_epoch': self.best_epoch,
            }
        )
        # architecture
        inputs, _ = next(iter(self.test_loader))
        self.writer.add_graph(self.model, inputs.to(self.device))

    def log_iter(self, iter_num, loss, metr):
        if iter_num % self.log_interval_iter != 0:
            return

        self.writer.add_scalar('Loss/train', loss, iter_num)
        self.writer.add_scalar('Metric/train', metr, iter_num)

        # norms of weights of convolution layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.writer.add_scalar(
                    f'Norm/Weight/{name}', torch.linalg.norm(module.weight), iter_num)
                self.writer.add_scalar(
                    f'Norm/Bias/{name}', torch.linalg.norm(module.bias), iter_num)
                if iter_num != 0:
                    self.writer.add_scalar(
                        f'Norm/grad_weight/{name}', torch.linalg.norm(module.weight.grad), iter_num)
                    self.writer.add_scalar(
                        f'Norm/grad_bias/{name}', torch.linalg.norm(module.bias.grad), iter_num)

    def log_gradients(self, epoch):
        input, target = self.pics_to_log_gradients

        # define hook function
        def hook(name, module, inp, out):
            out = out[-1]
            # for each object in batch
            for i in range(out.shape[0]):
                # visualize mean of feature maps
                img = out[i].mean(dim=0)
                img = img / img.max()
                self.writer.add_image(
                    f'Gradient[{i}]/module[{name}]', img,
                    epoch, dataformats='HW'
                )

        # register hooks
        handlers = []
        for name, layer in self.model.named_modules():
            is_enc_block = isinstance(layer, torch.nn.Sequential)
            is_dec_block = isinstance(layer, DecoderBlock)
            if not (is_enc_block or is_dec_block):
                continue

            layer.__name__ = name
            handler = layer.register_full_backward_hook(partial(hook, name))
            handlers.append(handler)

        # forward and backward pass
        self.model.eval()
        self.optimizer.zero_grad()
        preds = self.model(input.to(self.device))
        loss = self.criterion(preds, target.to(self.device)).mean()
        loss.backward()
        self.model.train()

        # disable hooks
        for handler in handlers:
            handler.remove()

    @torch.no_grad()
    def figure_to_log(
        self,
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    ):
        input, target = self.pics_to_log_demonstration
        input = input.to(self.device)
        target = target.to(self.device)

        # make predictions
        self.model.eval()
        logits = self.model(input.to(self.device))
        probs = torch.sigmoid(logits)
        pred_mask = probs >= 0.5

        # collect stats
        loss = self.criterion(logits, target).mean().item()
        metr = self.metric(logits, target).mean().item()

        n = len(input)
        fig, ax = plt.subplots(n, 4, figsize=(14, n*4))

        for i in range(n):
            pic = input[i]
            mask = target[i]
            prob = probs[i]
            pred = pred_mask[i]

            # watchable picture
            pic = pic.permute(1, 2, 0).cpu()
            pic = pic * std + mean
            pic = torch.clip(pic, min=0, max=1)

            # picture - probabilities - pred mask - true mask
            ax[i, 0].imshow(pic.cpu())
            ax[i, 1].imshow(prob.cpu().permute(1, 2, 0))
            ax[i, 2].imshow(pred.cpu().permute(1, 2, 0))
            ax[i, 3].imshow(mask.cpu().permute(1, 2, 0))

            ax[i, 0].set_title('original')
            ax[i, 1].set_title(f'predicted probs (loss {loss:.2f})')
            ax[i, 2].set_title(f'predicted mask (IoU {metr:.2f})')
            ax[i, 3].set_title('true mask')

        return fig
