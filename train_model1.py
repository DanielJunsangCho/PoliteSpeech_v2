from sklearn.model_selection import StratifiedKFold
import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from models.model1 import RNNTModel


def train(model, x_train, y_train, x_val, y_val):
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 100
    batch_size = 10
    batch_start = torch.arange(0, len(x_train), batch_size)

    best_acc = -np.inf
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:        #disable=False
            bar.set_desccription(f"Epoch {epoch}")
            for start in bar:
                x_batch = x_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                y_batch = y_batch.view(y_batch.size(dim=-1), 1)

                #forward pass
                y_pred = model(x_batch)
                loss = loss_func(y_pred, y_batch)

                #backward pass
                optimizer.zero_grad()
                loss.backward()

                #update weights
                optimizer.step()

                #print progress
                # acc = (y_pred.round() == y_batch).float().mean()
                # bar.set_postfix(
                #     loss=float(loss),
                #     acc=float(acc)
                # )
        model.eval()
        y_pred = model(x_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return best_acc

def cross_validation(x, y):
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    cv_scores = []
    for curr_train, curr_test in kfold.split(x, y):
        model = RNNTModel()
        acc = train(model, x[curr_train], y[curr_train], x[curr_test], y[curr_test])
        print("Accuracy (deep): $.3f" % acc)
        cv_scores.append(acc)


            

