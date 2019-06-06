import matplotlib.pyplot as plt

def plot_roi_centers(true, predict, model_name):
    plt.figure(figsize = (5, 5))
    plt.scatter(true[:, 0], true[:, 1], c='r', alpha=0.5, label='Human')
    plt.scatter(predict[:, 0], predict[:, 1], c='b', alpha=0.5, label=model_name)
    plt.axis('equal')
    plt.legend()
    plt.title("ROI-centers of randomly selected images")
    plt.show() 

def plot_history(hist_model, model_name, xlim=None, ylim=None):
    loss_list = [s for s in hist_model.history.keys() 
                 if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in hist_model.history.keys()
                     if 'loss' in s and 'val' in s]

    
    assert len(loss_list) > 0, "Loss is missing in history."
    
    ## As loss always exists
    epochs = range(1, len(hist_model.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure()
    for l in loss_list:
        plt.plot(epochs, hist_model.history[l], 'b',
                 label = "Training ({:.5f})".format(hist_model.history[l][-1]))
    for l in val_loss_list:
        plt.plot(epochs, hist_model.history[l], 'g', 
                 label = "Validation ({:.5f})".format(hist_model.history[l][-1]))

    if xlim is not None:
        plt.xlim(xlim)    
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Euclidean Distance')
    plt.legend()
    plt.grid()
    
    plt.show()