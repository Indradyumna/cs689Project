import torch
import matplotlib.pyplot as plt


def plot_data (epoch, tr_acc_list,val_acc_list):
    epochs_range = list(range(epoch))
    plt.plot(epochs_range, tr_acc_list, label='Training Loss')
    plt.plot(epochs_range, val_acc_list, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_data1 (epoch, tr_loss_list):
    epochs_range = list(range(epoch))
    plt.plot(epochs_range, tr_loss_list, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def train(av, loader_train, loader_valid, model):
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=av.LEARNING_RATE, weight_decay=0.0)
    stop_early = False
    num_bad_epochs = 0
    patience = 50
    best_acc= 0
    best_val_model = model
    
    epoch = 0

    tr_acc_list = [] 
    val_acc_list = [] 
    while (not stop_early):
        epoch_loss = 0
        for batch_data in loader_train:
            batch_data.to("cuda")

            out = model(batch_data)
            loss = torch.nn.NLLLoss()(out, batch_data.y.flatten())
            epoch_loss = epoch_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tr_acc = evaluate_accuracy(model, loader_train)
        val_acc = evaluate_accuracy(model, loader_valid)
        tr_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)

        print("EpochLoss: {:.3f} TrAcc: {:.3f} ValAcc: {:.3f} ".format(epoch_loss,tr_acc,val_acc))

        epoch = epoch + 1
        if epoch%25 == 0:
            plot_data (epoch, tr_acc_list,val_acc_list)
        if val_acc > best_acc: 
            best_acc = val_acc 
            num_bad_epochs = 0
            best_val_model = model
            torch.save(best_val_model.state_dict(), av.DIR_PATH+"/best_val_models/"+av.DATASET_NAME+\
                       '_'+ av.CONV+'_'+str(av.LEARNING_RATE))
        else: 
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs > patience: 
                stop_early = True
        print("bestAcc: {:.3f} numBepochs: {:d}")
    plot_data (epoch, tr_acc_list,val_acc_list)

    
def train_explainer(av, model_exp, test_data):
    model_exp.model.eval()
    model_exp.clear_mask()
    model_exp.init_mask()
    test_data.to("cuda")

    optimizer = torch.optim.Adam(params=[model_exp.node_feat_mask, model_exp.edge_mask],\
                                 lr=0.01, weight_decay=0.0)
    stop_early = False
    num_bad_epochs = 0
    patience = 50
    best_loss= 100000
    delta = 0.0001
    best_val_model = model_exp
    
    epoch = 0

    tr_loss_list = [] 
    while (not stop_early):
        d1  = test_data.detach().clone()
        loss = model_exp(d1.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss_list.append(loss.item())

        #print("TrLoss: {:.6f} ".format(loss))

        epoch = epoch + 1
        if epoch%500 == 0:
            plot_data1 (epoch, tr_loss_list)
        if loss < best_loss-delta: 
            best_loss = loss
            num_bad_epochs = 0
            best_val_model = model_exp
        else: 
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs > patience: 
                stop_early = True
    plot_data1 (epoch, tr_loss_list)
    
    
    node_feat_mask = best_val_model.node_feat_mask.detach().sigmoid()
    edge_mask = best_val_model.edge_mask.detach().sigmoid()
    model_exp.clear_mask()

    return node_feat_mask, edge_mask


def evaluate_accuracy(model, loader): 
    """
    """
    model.eval()

    all_pred = []
    label = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to("cuda")
        all_pred.append(model(data).argmax(dim=1))  
        label.append(data.y) 
    all_pred = torch.cat(all_pred).flatten()
    label = torch.cat(label).flatten()
    acc = torch.sum(all_pred == label).item() / len(all_pred) 
    return acc


def cudavar(av, x):
    """Adapt to CUDA or CUDA-less runs.  Annoying av arg may become
    useful for multi-GPU settings."""
    return x.cuda() if av.has_cuda and av.want_cuda else x
