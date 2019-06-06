import matplotlib.pyplot as plt
import cv2
import math

def draw_roi(img_path, img_list, true_centers, pred_centers=None, 
             actual_R=1.5, actual_imgsize=16.38, current_resol=1330, 
             org_resol=2048, rows=1, cols=3, model_name="Model"):
    
    resize_ratio = current_resol / org_resol
    R = math.floor(current_resol * actual_R/(actual_imgsize*resize_ratio))
    
    plt.figure(figsize=(8*cols, 8*rows))
    
    for i, img_id in enumerate(img_list):
        plt.subplot(rows, cols, i+1)
        img = cv2.imread(img_path + img_id + '.png')
        cx = math.floor(true_centers[i, 0] * current_resol)
        cy = math.floor(true_centers[i, 1] * current_resol)        
        
        if pred_centers is not None:
            cx_pred = math.floor(pred_centers[i, 0] * current_resol) 
            cy_pred = math.floor(pred_centers[i, 1] * current_resol)
            img_mod = cv2.circle(img, (cx_pred, cy_pred), R, (0,0,255), 3)  # Mark the predicted center in blue
            img_mod = cv2.circle(img_mod, (cx_pred, cy_pred), round(R*0.05), (0,0,255), -1)  
            img_mod = cv2.circle(img_mod, (cx, cy), R, (255,0,0), 2)  # Mark the true center in red
            img_mod = cv2.circle(img_mod, (cx, cy), round(R*0.05), (255,0,0), -1)  

        else:
            img_mod = cv2.circle(img, (cx, cy), R, (255,0,0), 3)  # Mark the true center in red
            img_mod = cv2.circle(img_mod, (cx, cy), round(R*0.05), (255,0,0), -1)  

        plt.imshow(img_mod)
    
        if pred_centers is not None:
            plt.title("ROI center\n" + model_name + ": {}\nHuman: {}"
                      .format(str((cx_pred, cy_pred)),
                              str((cx, cy))))
        else:
            plt.title("ROI center: {}".format(str((cx, cy))))
            
            
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
    rmse_list = [s for s in hist_model.history.keys() 
                 if 'rmse' in s and 'val' not in s]
    val_rmse_list = [s for s in hist_model.history.keys()
                     if 'rmse' in s and 'val' in s]
    
    assert len(loss_list) > 0, "Loss is missing in history."
    
    ## As loss always exists
    epochs = range(1, len(hist_model.history[loss_list[0]]) + 1)
    
    fig = plt.figure(figsize=(10, 5))
    ## Plot loss
    ax1 = plt.subplot(1,2,1)
    for l in loss_list:
        ax1.plot(epochs, hist_model.history[l], 'b',
                 label = "Training ({:.5f})".format(hist_model.history[l][-1]))
    for l in val_loss_list:
        ax1.plot(epochs, hist_model.history[l], 'g', 
                 label = "Validation ({:.5f})".format(hist_model.history[l][-1]))

    if xlim is not None:
        plt.xlim(xlim)    
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Epochs')
    plt.ylabel('Mean L2 Error')
    plt.legend()
    plt.grid()
    
    ## Plot RMSE
    if len(rmse_list) > 0:
        ax2 = plt.subplot(1,2,2, sharey=ax1)
        for l in rmse_list:
            ax2.plot(epochs, hist_model.history[l], 'b',
                     label = "Training ({:.5f})".format(hist_model.history[l][-1]))
        for l in val_rmse_list:
            ax2.plot(epochs, hist_model.history[l], 'g', 
                     label = "Validation ({:.5f})".format(hist_model.history[l][-1]))
    
        if xlim is not None:
            plt.xlim(xlim)    
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel('Epochs')
        plt.ylabel('Root Mean Square Error')
        plt.legend()
        plt.grid()
    
    
    fig.suptitle(model_name, fontsize=14)
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.show()