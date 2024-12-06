#%%
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def evaluate_model(model, dataloader, setting, savePATH):
    model.eval()
    with torch.no_grad():
        for idxVal, batch in enumerate(dataloader):
            x_NLL, z_KLD, a_NLL, Readout, Target, z_Sample, a_Sample, h_Sample, zh_Sample = model(batch, setting, sample=True)
            y_BCE, y_ACC, y_Prob, y_Pred = model.classifier(Readout, Target, sample=True)
            total_loss = x_NLL + z_KLD + a_NLL + y_BCE

    return y_BCE, y_ACC, y_Pred, Target, z_Sample, zh_Sample, Readout

def plot_losses(checkpoint, savePATH):
    plt.plot(torch.stack(checkpoint['train_losses']['z_KLD'], dim=0).cpu().numpy())
    plt.plot(torch.stack(checkpoint['test_losses']['z_KLD'], dim=0).cpu().numpy())
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(torch.stack(checkpoint['train_losses']['a_NLL'], dim=0).cpu().numpy())
    plt.plot(torch.stack(checkpoint['test_losses']['a_NLL'], dim=0).cpu().numpy())
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(torch.stack(checkpoint['train_losses']['y_BCE'], dim=0).cpu().numpy())
    plt.plot(torch.stack(checkpoint['test_losses']['y_BCE'], dim=0).cpu().numpy())
    plt.show()

    plt.plot(torch.stack(checkpoint['train_losses']['y_ACC'], dim=0).cpu().numpy())
    plt.plot(torch.stack(checkpoint['test_losses']['y_ACC'], dim=0).cpu().numpy())
    plt.show()

def plot_tsne(z_Samples, Targets, testIdx, title, saveTo):
    tsne = TSNE(n_components=2)
    Target = torch.cat(Targets).detach().cpu().numpy()
    hc_idx = (Target == 0)
    asd_idx = (Target == 1)

    test_idx = torch.cat(testIdx).detach().cpu().numpy()
    train_idx = (test_idx == 0)
    test_idx = (test_idx == 1)

    samples = [item.detach().cpu().numpy() for sublist in z_Samples for item in sublist]
    data_fit = [sample.reshape(len(sample), -1) for sample in samples]
    sub_len = [len(data) for data in data_fit]
    split_idx = np.cumsum(sub_len)
    tsne_embed = tsne.fit_transform(np.concatenate(data_fit, axis=0))
    z_embed = np.split(tsne_embed, split_idx, axis=0)[:-1]

    z_embed_hc_train = np.concatenate([z_embed[i] for i in np.where(hc_idx * train_idx)[0]], axis=0)
    z_embed_asd_train = np.concatenate([z_embed[i] for i in np.where(asd_idx * train_idx)[0]], axis=0)
    z_embed_hc_test = np.concatenate([z_embed[i] for i in np.where(hc_idx * test_idx)[0]], axis=0)
    z_embed_asd_test = np.concatenate([z_embed[i] for i in np.where(asd_idx * test_idx)[0]], axis=0)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.scatter(z_embed_hc_train[:, 0], z_embed_hc_train[:, 1], c='blue', marker='.', s=40, linewidth=1)
    ax.scatter(z_embed_asd_train[:, 0], z_embed_asd_train[:, 1], c='blue', marker='x', s=40, linewidth=1)
    ax.scatter(z_embed_hc_test[:, 0], z_embed_hc_test[:, 1], c='red', marker='.', s=40, linewidth=1)
    ax.scatter(z_embed_asd_test[:, 0], z_embed_asd_test[:, 1], c='red', marker='x', s=40, linewidth=1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    os.makedirs(saveTo, exist_ok=True)
    plt.savefig(saveTo + title + '_Z_default.jpg', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.show()