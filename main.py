import torch
from sklearn.metrics import f1_score
from utils import load_data, EarlyStopping
from model import HGSAGE
import pandas as pd


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func, args, movie_nodes=None):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, args)
    if movie_nodes is not None:
        logits = torch.index_select(logits, 0, movie_nodes)
        loss = loss_func(logits[mask], labels[mask])
    else:
        loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    if args['dataset'] == 'IMDB':
        g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
        val_mask, test_mask, movie_nodes = load_data(args['dataset'])
        movie_nodes = movie_nodes.to(args['device'])
    else:
        g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
        val_mask, test_mask = load_data(args['dataset'])

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()  # 布尔类型转换
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    model = HGSAGE(meta_paths=[['ma', 'am'], ['md', 'dm']],    # 之前构建的边: pa, ap,组合成meta-path: PAP
                   in_size=features.shape[1],
                   hidden_size=args['hidden_units'],
                   out_size=num_classes,
                   dropout=args['dropout'],
                   aggregator_type=args['aggregator_type']).to(args['device'])
    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features, args)
        if args['dataset'] == 'IMDB':
            logits = torch.index_select(logits, 0, movie_nodes)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        else:
            loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        if args['dataset'] == 'IMDB':
            val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn,
                                                                     args, movie_nodes)
        else:
            val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn,
                                                                     args)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    if args['dataset'] == 'IMDB':
        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn,
                                                                     args, movie_nodes)
    else:
        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn,
                                                                     args)
    embeddings = pd.read_csv(f"{args['dataset']}_embeddings_{args['aggregator_type']}+g.csv", header=None)

    if args['dataset'] == 'IMDB':
        embeddings = embeddings.iloc[movie_nodes.cpu(), :]
    embeddings.insert(loc=len(embeddings.columns), column=len(embeddings.columns), value=labels.cpu().numpy())
    embeddings.to_csv(f"{args['dataset']}_embeddings_{args['aggregator_type']}+g.csv", index=False, header=False)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))


if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser(description='HGSAGE')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')

    args = parser.parse_args().__dict__
    args = setup(args)

    main(args)


