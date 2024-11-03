
"""

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.lin1 = nn.Linear(dataset.num_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=args.drop, training=self.training)
        y = self.lin2(x1)  # acc = 0.55
        # y = self.lin2(x1+x)
        return y



def train_mlp():
    data = load_data()
    model = MLP(hidden_channels=args.hidden, dataset=data)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # Define optimizer.

    def train():
        model.train()
        model.to(device=_device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def eva():
        model.eval()
        out = model(data.x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    max_acc = -1
    records = []
    for cur_epoch in range(1, 100):
        loss = train()
        test_acc = eva()
        if (max_acc < test_acc):
            max_acc = test_acc
            if (max_acc > args.min_acc):
                save_model(model=model, **vars(args))
        if (cur_epoch % 30 == 0):
            print_loss(epoch=cur_epoch, loss=loss.item(), test_acc=test_acc)
            records.append({'epoch': cur_epoch, 'loss': loss.item(), 'test_acc': test_acc})
    args.max_acc = max_acc
    save_json(records=records, **vars(args))
    print(max_acc)

"""