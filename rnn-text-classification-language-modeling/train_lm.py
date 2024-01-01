import torch
from lstm.model_clf import LMAccuracy


def get_and_send(batch, device):
  for key in ['tokens', 'tokens_lens']:
    yield batch[key].to(device)

def train_epoch_lm(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for idx, batch in enumerate(dataloader):
        tokens, tokens_lens = get_and_send(batch, device)
        model.zero_grad()
        output = model(tokens, tokens_lens)
        loss = loss_fn(output, tokens, tokens_lens)
        loss.backward()
        optimizer.step()
    
def evaluate_lm(dataloader, model, loss_fn, device):
    model.eval()
    
    total_tokens = 0
    total_loss = 0.0
    total_accuracy = 0.0
    
    accuracy_fn = LMAccuracy()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            tokens, tokens_lens = get_and_send(batch, device)
            output = model(tokens, tokens_lens)
            loss = loss_fn(output, tokens, tokens_lens).item()
            acc = accuracy_fn(output, tokens, tokens_lens).item()
            n_tokens = (tokens_lens + 1).sum().item()
            total_loss += loss * n_tokens
            total_accuracy += acc
            total_tokens += n_tokens
            
    return total_loss / total_tokens, total_accuracy / total_tokens

def train_lm(
    train_loader, test_loader, model, loss_fn, optimizer, device, num_epochs
):
    test_losses = []
    train_losses = []
    test_accuracies = []
    train_accuracies = []
    for epoch in range(num_epochs):
        train_epoch_lm(train_loader, model, loss_fn, optimizer, device)
        
        train_loss, train_acc = evaluate_lm(train_loader, model, loss_fn, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        test_loss, test_acc = evaluate_lm(test_loader, model, loss_fn, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        print(
            'Epoch: {0:d}/{1:d}. Loss (Train/Test): {2:.3f}/{3:.3f}. Accuracy (Train/Test): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]
            )
        )
    return train_losses, train_accuracies, test_losses, test_accuracies