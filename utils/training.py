import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.optim.lr_scheduler import MultiStepLR
from e3nn.nn import Dropout
from utils.model import PeriodicNetwork, PeriodicNetworkPhdos
import numpy as np
import time
bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.region = 1
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        if self.region == 1:
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.region = 2
                    self.counter = 0
                    self.patience = 500
            else:

                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        elif self.region == 2:

            if score > self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.region = 3
                    self.counter = 0
                    self.patience = 5000



            else:

                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        else:

            if score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:

                    self.early_stop = True


            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',flush=True)
        torch.save(model.state_dict(), self.path+ f'{self.region}'+'.pt')
        self.val_loss_min = val_loss




def train(
    model,
    optimizer,
    dataloader_train,
    dataloader_valid,
    loss_fn,
    loss_fn_mae,
    run_name,
    max_iter=101,
    scheduler=None,
    device="cpu",
):
    model.to(device)
    early_stopping = EarlyStopping(path=run_name+'.pt',patience=15, verbose=True)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    try:
        model.load_state_dict(torch.load(run_name + ".torch")["state"])
    except:
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + ".torch")
        history = results["history"]
        s0 = history[-1]["step"] + 1
    best_error = 10e10

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.0
        loss_cumulative_mae = 0.0

        for j, d in enumerate(dataloader_train):
            d.to(device)
            output = model(d)
            loss = loss_fn(
                output, d.target
            )  # / torch.Tensor(np.arange(0.25,101,2.))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mae = loss_fn_mae(output, d.target).cpu()
            loss_cumulative = loss_cumulative + loss.cpu().detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
            
        end_time = time.time()
        wall = end_time - start_time

        valid_avg_loss = evaluate(
            model, dataloader_valid, loss_fn, loss_fn_mae, device
        )
        train_avg_loss = evaluate(
            model, dataloader_train, loss_fn, loss_fn_mae, device
        )

        history.append(
            {
                "step": s0 + step,
                "wall": wall,
                "batch": {"loss": loss.item(), "mean_abs": loss_mae.item()},
                "valid": {"loss": valid_avg_loss[0], "mean_abs": valid_avg_loss[1]},
                "train": {"loss": train_avg_loss[0], "mean_abs": train_avg_loss[1]},
            }
        )

        results = {"history": history, "state": model.state_dict()}
        if (step+1)%100 == 0:
            print(
                f"Iteration {step+1:4d}   "
                + f"train loss = {train_avg_loss[0]:8.6f}   "
                + f"valid loss = {valid_avg_loss[0]:8.6f}   "
                + f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}",
                flush=True,
            )
        if step%5 == 0:
            with open(run_name + ".torch", "wb") as f:
                torch.save(results, f)

        early_stopping(valid_avg_loss[0], model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if scheduler is not None:
            scheduler.step()


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, dataloader, loss_fn, loss_fn_mae, device):
    model.eval()
    loss_cumulative = 0.0
    loss_cumulative_mae = 0.0
    start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output = model(d)

            loss = loss_fn(output, d.target).cpu()

            # loss = loss_fn(output, d.target).cpu()
            loss_mae = loss_fn_mae(output, d.target).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    return loss_cumulative / len(dataloader), loss_cumulative_mae / len(dataloader)        

def get_model(init_dict,lr = 0.005, wd=0.05,total=False,device='cpu'):
    if total:
        model = PeriodicNetworkPhdos(**init_dict)
    else:
        model = PeriodicNetwork(**init_dict)
        model.pool = True
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = MultiStepLR(opt, milestones=[900001],gamma=0.1)

    return model, opt, scheduler
