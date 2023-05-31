import torch


def get_and_send(batch, device):
    for key in ['tokens', 'tokens_lens', 'ratings']:
        yield batch[key].to(device)


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for idx, batch in enumerate(dataloader):
        # batch is dict in which we are interested
        # in keys 'tokens', 'tokens_lens', 'ratings'
        tokens, tokens_lens, ratings = get_and_send(batch, device)

        # forward pass
        model.zero_grad()
        output = model(tokens, tokens_lens)
        loss = loss_fn(output, ratings)

        # compute gradients
        loss.backward()

        # update net params
        optimizer.step()


def evaluate(dataloader, model, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # batch is dict in which we are interested
            # in keys 'tokens', 'tokens_lens', 'ratings'
            tokens, tokens_lens, ratings = get_and_send(batch, device)

            # forward pass
            output = model(tokens, tokens_lens)

            # collect stats
            total_loss += loss_fn(output, ratings).item() * len(ratings)
            total_accuracy += (output.argmax(axis=1) == ratings).sum().item()

    # average values
    return total_loss / len(dataloader.dataset), total_accuracy / len(dataloader.dataset)


def train(
    train_loader, test_loader, model, loss_fn, optimizer, device, num_epochs
):
    test_losses = []
    train_losses = []
    test_accuracies = []
    train_accuracies = []
    for epoch in range(num_epochs):
        train_epoch(train_loader, model, loss_fn, optimizer, device)

        train_loss, train_acc = evaluate(train_loader, model, loss_fn, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        print(
            f'Epoch: {epoch + 1}/{num_epochs}.',
            f'Loss (Train/Test): {train_losses[-1]:.3f}/{test_losses[-1]:.3f}.',
            f'Accuracy (Train/Test): {train_accuracies[-1]:.3f}/{test_accuracies[-1]:.3f}',
            sep='\t'
        )
    return train_losses, train_accuracies, test_losses, test_accuracies
