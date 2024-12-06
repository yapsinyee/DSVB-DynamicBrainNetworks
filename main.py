#%%
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from model import *
from train import *
from evaluate import *

outer_loop = 1; inner_loop = 2
train_graphs, test_graphs, val_graphs = load_data(outer_loop, inner_loop)

train_dataset = myDataset(np.concatenate([train_graphs, val_graphs], axis=0))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)

val_dataset = myDataset(val_graphs)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)

test_dataset = myDataset(test_graphs)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)

partition = [len(train_dataset), len(val_dataset), len(test_dataset)]
print(len(train_graphs), len(val_graphs), len(test_graphs))
print(partition)

#%% Initialize training
savePATH = 'saved_models/VGRNN_softmax_adv_fold'+str(outer_loop)+str(inner_loop)
loadPATH = savePATH

model_params = {'num_nodes':264, 'num_classes':2, 'x_dim':264, 'y_dim':2,
				'z_hidden_dim':32, 'z_dim':16, 'z_phi_dim':8,
				'x_phi_dim':64, 'rnn_dim':16, 'y_hidden_dim':[32],
				'x_hidden_dim':64, 'layer_dims':[]
				}

# 4096,2048,1024,512,256,128,64,32,16,8,4,2

lr_annealType = 'ReduceLROnPlateau'
lr_annealType = [lr_annealType, lr_annealType]

setting = {
'rngPATH': r"saved_models/VGRNN_softmax_adv_fold11",
'model_params': model_params,
'recurrent': True,
'learnRate': [0.00001, 0.00001],
'yBCEMultiplier': [1, 1],
'l2factor': [0.01, 0.01],
'lr_annealType': lr_annealType,
'lr_annealFactor': [0.5, 0.5],
'lr_annealPatience': [20, 20],
'variational': True,
'DAT': False,
'graphRNN': True,
'partition': partition
}

# Acquire model, optimizer and scheduler
model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses = loadCheckpoint(setting, loadPATH, savePATH)
print(model)
print(model_params)
for optimizer in optimizers:
	print(optimizer)
for scheduler in schedulers:
	print(scheduler)
print(setting['rngPATH'])
print(savePATH)


# %% Train model
model, train_losses, val_losses, test_losses = train(
model, optimizers, schedulers, setting, savePATH,
train_losses, val_losses, test_losses,
train_loader, val_loader, test_loader, 
epochStart=epochStart, numEpochs=10, 
gradThreshold=1, gradientClip=True,
verboseFreq=1, verbose=True, valFreq=1, 
validation=False, testing=True,
earlyStopPatience=500, earlyStop=True)


# %% Check the training process
savePATH = "saved_models/VGRNN_softmax_adv_fold12"

loadPATH = savePATH
checkpoint = torch.load(savePATH)
setting = checkpoint['training_setting']

plt.plot(torch.stack(checkpoint['train_losses']['z_KLD'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['z_KLD'], dim=0).cpu().numpy())
plt.legend(['train','test'])
plt.title('z_KLD Losses')
plt.show()

plt.plot(torch.stack(checkpoint['train_losses']['a_NLL'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['a_NLL'], dim=0).cpu().numpy())
plt.legend(['train','test'])
plt.title('a_NLL Losses')
plt.show()

plt.plot(torch.stack(checkpoint['train_losses']['y_BCE'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['y_BCE'], dim=0).cpu().numpy())
plt.legend(['train','test'])
plt.title('y_BCE Losses')
plt.show()

plt.plot(torch.stack(checkpoint['train_losses']['y_ACC'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['y_ACC'], dim=0).cpu().numpy())
plt.legend(['train','test'])
plt.title('y_ACC Losses')
plt.show()

print("savePATH:", savePATH)
print("training_setting:", checkpoint['training_setting'])
print("test_losses y_BCE:", checkpoint['test_losses']['y_BCE'][-1], 
      "test_losses y_ACC:", 1 - checkpoint['test_losses']['y_ACC'][-1])


#%% Evaluation
savePATHS = []; test_ACC = []; test_PRED = []; test_SENS = []; test_F1 = []; test_AUC = []
z_Samples = []; zh_Samples = []; Readouts = []; Targets = []; testIdx = []

savePATHS.append(r"saved_models/VGRNN_softmax_adv_fold11")
savePATHS.append(r"saved_models/VGRNN_softmax_adv_fold12")

for savePATH in savePATHS:

	# Extract the part that contains fold
	fold_part = savePATH.split('_')[-1]

	# Extract the digits from fold
	outer_loop = fold_part[-2]
	inner_loop = fold_part[-1]
	train_graphs, test_graphs, val_graphs = load_data(outer_loop, inner_loop)

	dataset = myDataset(np.concatenate([train_graphs, val_graphs, test_graphs], axis=0))
	# dataset = myDataset(np.concatenate([test_graphs], axis=0))
	dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, collate_fn=padseq, pin_memory=True)

	test_idx = torch.zeros(len(dataset))
	test_idx[:len(test_graphs)] = True

	loadPATH = savePATH
	checkpoint = torch.load(savePATH)
	setting = checkpoint['training_setting']
	model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses = loadCheckpoint(setting, loadPATH, savePATH)

	for idxVal, batch in enumerate(dataloader):

		# Compute validation batch losses (and generate validation samples)
		model.eval()
		with torch.no_grad():
			x_NLL, z_KLD, a_NLL, Readout, Target, z_Sample, a_Sample, h_Sample, zh_Sample = model(batch, setting, sample=True)
			y_BCE, y_ACC, y_Prob, y_Pred = model.classifier(Readout, Target, sample=True)
			total_loss = x_NLL + z_KLD + a_NLL + y_BCE

	print('outer_loop: '+ str(outer_loop))
	print('savePATH: '+ savePATH)
	print("| recurrent:", setting['recurrent'] if 'recurrent' in setting else True, 
      "| variational:", setting['variational'], 
      "| DAT:", setting['DAT'], 
      "| graphRNN:", setting['graphRNN'] if 'graphRNN' in setting else True)

	z_Samples.append(z_Sample)
	zh_Samples.append(zh_Sample)
	Readouts.append(Readout)
	Targets.append(Target)
	testIdx.append(test_idx)

	Pred = y_Pred.detach().cpu().numpy()
	Actual = Target.detach().cpu().numpy()
	# accuracy
	test_acc = np.sum(Pred==Actual)/len(Pred)
	test_ACC.append(test_acc)
	# precision
	test_pred = precision_score(Actual, Pred)
	test_PRED.append(test_pred)
	# recall
	test_sens = recall_score(Actual, Pred)
	test_SENS.append(test_sens)
	# F1-score
	test_f1 = f1_score(Actual, Pred)
	test_F1.append(test_f1)
	# AUC
	test_auc = roc_auc_score(Actual, Pred)
	test_AUC.append(test_auc)
	print('Inner Loop:', inner_loop)
	print(test_acc, test_pred, test_sens, test_f1, test_auc)

print(f"===== Outer Loop: {outer_loop} =====")
print(f"Accuracy: {np.mean(test_ACC):.4f}, Std: {np.std(test_ACC):.4f}")
print(f"Precision: {np.mean(test_PRED):.4f}, Std: {np.std(test_PRED):.4f}")
print(f"Sensitivity: {np.mean(test_SENS):.4f}, Std: {np.std(test_SENS):.4f}")
print(f"F1 Score: {np.mean(test_F1):.4f}, Std: {np.std(test_F1):.4f}")
print(f"AUC: {np.mean(test_AUC):.4f}, Std: {np.std(test_AUC):.4f}")
print("===================================")


#%% Extract the embeddings for one outer loop
saveTo = './outputs/samples_data/'  
os.makedirs(saveTo, exist_ok=True)

samples = {}
samples['PATH'] = savePATH
samples['metrics'] = (y_BCE.detach().cpu().numpy(), y_ACC.detach().cpu().numpy())
samples['latent'] = [sample.detach().cpu().numpy() for sample in z_Sample]
samples['adjacency'] = [sample.detach().cpu().numpy() for sample in a_Sample]
samples['recurrent'] = [sample.detach().cpu().numpy() for sample in h_Sample]
samples['embedding'] = [sample.detach().cpu().numpy() for sample in zh_Sample]
samples['target'] = Target.detach().cpu().numpy()
samples['prob'] = y_Prob.detach().cpu().numpy()
samples['pred'] = y_Pred.detach().cpu().numpy()

print('Saving samples ...')     
with open(saveTo+'samples_outer'+str(outer_loop)+'.pkl', 'wb') as f:
	torch.save(samples, f)
	f.close()

#%% Plot tSNE
plt.style.use('default')
plt.style.use('Solarize_Light2')
init = 'default'
tsne = TSNE(n_components=2)

Target = torch.cat(Targets).detach().cpu().numpy()
hc_idx = (Target == 0)
asd_idx = (Target == 1)

test_idx = torch.cat(testIdx).detach().cpu().numpy()
train_idx = (test_idx == 0)
test_idx = (test_idx == 1)

samples = [item.detach().cpu().numpy() for sublist in z_Samples for item in sublist]
data_fit = [sample.reshape(len(sample),-1) for sample in samples]
sub_len = [len(data) for data in data_fit]
split_idx = np.cumsum(sub_len)
tsne_embed = tsne.fit_transform(np.concatenate(data_fit, axis=0))
z_embed = np.split(tsne_embed, split_idx, axis=0)[:-1]
print(sub_len == [len(item) for item in z_embed])
z_embed_hc_train = np.concatenate([z_embed[i] for i in np.where(hc_idx*train_idx)[0]], axis=0)
z_embed_asd_train = np.concatenate([z_embed[i] for i in np.where(asd_idx*train_idx)[0]], axis=0)
z_embed_hc_test = np.concatenate([z_embed[i] for i in np.where(hc_idx*test_idx)[0]], axis=0)
z_embed_asd_test = np.concatenate([z_embed[i] for i in np.where(asd_idx*test_idx)[0]], axis=0)

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
ax.scatter(z_embed_hc_train[:,0], z_embed_hc_train[:,1], c='blue', marker='.', s=40, linewidth=1)
ax.scatter(z_embed_asd_train[:,0], z_embed_asd_train[:,1], c='blue', marker='x', s=40, linewidth=1)
ax.scatter(z_embed_hc_test[:,0], z_embed_hc_test[:,1], c='red', marker='.', s=40, linewidth=1)
ax.scatter(z_embed_asd_test[:,0], z_embed_asd_test[:,1], c='red', marker='x', s=40, linewidth=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Readout = torch.cat(Readouts).detach().cpu().numpy()
# r_embed = tsne.fit_transform(Readout)
# r_embed_hc_train = r_embed[hc_idx*train_idx]
# r_embed_asd_train = r_embed[asd_idx*train_idx]
# r_embed_hc_test = r_embed[hc_idx*test_idx]
# r_embed_asd_test = r_embed[asd_idx*test_idx]

# fig = plt.figure(figsize=(10,10))
# ax = plt.axes()
# ax.scatter(r_embed_hc_train[:,0], r_embed_hc_train[:,1], c='blue', marker='.', s=80, linewidth=2)
# ax.scatter(r_embed_asd_train[:,0], r_embed_asd_train[:,1], c='blue', marker='x', s=80, linewidth=2)
# ax.scatter(r_embed_hc_test[:,0], r_embed_hc_test[:,1], c='red', marker='.', s=80, linewidth=2)
# ax.scatter(r_embed_asd_test[:,0], r_embed_asd_test[:,1], c='red', marker='x', s=80, linewidth=2)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

saveTo = './outputs/saved_figures/'  
os.makedirs(saveTo, exist_ok=True)
title = "tSNE_z"
# plt.title(title)
# plt.legend(['HC train','ASD train','HC test','ASD test'], loc='best', ncol=1, columnspacing=0.3, handletextpad=0.3, borderpad=0.2, fontsize=20)
plt.savefig(saveTo+title+'_Z'+'_'+init+'.jpg', dpi=400, bbox_inches='tight', pad_inches=0)
plt.show()

# %%
