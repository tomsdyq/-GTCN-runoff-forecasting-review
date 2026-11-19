"""Training and evaluation routines for the G-TCN release."""
import numpy as np
import torch


def forecast(model, batch, device):
    weather, history, target, scale = batch
    weather, history, target = [v.to(device) for v in (weather, history, target)]
    runoff = torch.cat((history, torch.zeros_like(target)), dim=1)
    prediction = model(torch.cat((weather, runoff), dim=2))[:, -target.size(1):]
    return prediction, target, scale


def train_one_epoch(model, loader, optimizer, regional, device):
    model.train()
    total = 0.0
    for batch in loader:
        pred, target, scale = forecast(model, batch, device)
        errors = (pred - target).square()
        if regional:
            weights = (scale.to(device) + 0.1).pow(-2).view(-1, 1, 1)
            errors = errors * weights
        loss = errors.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validation(model, loader, device):
    model.eval()
    mses, scores = [], []
    for batch in loader:
        pred, target, _ = forecast(model, batch, device)
        error = (pred - target).square()
        # Historical checkpoint selection used the variance of predictions.
        denom = (pred - pred.mean(dim=0)).square().sum(dim=0)
        scores.append((1 - error.sum(dim=0) / denom).mean().item())
        mses.append(error.mean().item())
    return np.mean(mses), np.mean(scores)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    targets, predictions = [], []
    for batch in loader:
        pred, target, _ = forecast(model, batch, device)
        targets.append(target.cpu())
        predictions.append(pred.cpu())
    return torch.cat(targets), torch.cat(predictions)


def test_scores(observed, predicted):
    y, p = observed[:, :, 0], predicted[:, :, 0]
    error = p - y
    mean_y, mean_p = y.mean(0), p.mean(0)
    var_y, var_p = y.std(0), p.std(0)
    corr = ((p - mean_p) * (y - mean_y)).mean(0) / (var_p * var_y)
    count = max(1, int(len(y) * 0.02))
    peaks = np.argsort(y, axis=0)[-count:]
    peak_error = np.take_along_axis(abs(error), peaks, axis=0)
    peak_flow = np.take_along_axis(y, peaks, axis=0)
    return {
        "NSE": 1 - (error**2).sum(0) / ((y - mean_y)**2).sum(0),
        "KGE": 1 - np.sqrt((corr - 1)**2 + (var_p / var_y - 1)**2 + (mean_p / mean_y - 1)**2),
        "RMSE": np.sqrt((error**2).mean(0)), "Bias": error.sum(0) / y.sum(0),
        "ATPE2": peak_error.sum(0) / peak_flow.sum(0),
    }

