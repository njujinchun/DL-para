import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch as th
plt.switch_backend('agg')


def load_data(hdf5_dir, opt):

    with h5py.File(hdf5_dir + "/input_k_{}.hdf5".format(opt.n_data), 'r') as f:
        x = f['dataset'][()]
        x_train= x[:opt.n_train]        
        x_test= x[-opt.n_test:]    
    print("total training data shape: {}".format(x_train.shape))
    print("total testing data shape: {}".format(x_test.shape))
    
    n_out_pixels = np.prod(x_test.shape)  
    data = th.utils.data.TensorDataset(th.FloatTensor(x_train))
    data_loader = th.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=int(2))

    return data_loader, x_test, n_out_pixels


def plot_pred(samples, epoch, idx, output_dir):
    Ncol = 10
    
    fig, axes = plt.subplots(3, Ncol, figsize=(Ncol*3, 8))
    fs = 14 # font size
    title = (['$Layer1$','$Layer2$','$Layer3$','$Layer4$','$Layer5$','$Layer6$','$Layer7$','$Layer8$','$Layer9$','$Layer10$'])
    ylabel = (['$\mathbf{Original}$', '$\mathbf{Reconstructed}$', '$\mathbf{Random}$'])
    k = 0    
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')        
        ax.set_xticks([])
        ax.set_yticks([])
        if j%Ncol != 9:
            ax.imshow(samples[j], cmap='jet', origin='lower', vmin=-20, vmax=-11)
                    
        if j%Ncol == 9:

            cax = ax.imshow(samples[j], cmap='jet', origin='lower', vmin=-20, vmax=-11)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.025, pad=0.015)                          
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
            cbar.ax.tick_params(axis='both', which='both', length=0)            
            cbar.ax.tick_params(labelsize=fs-2)                                                
              
        if j < Ncol:
            ax.set_title(title[j],fontsize=fs-2)

        if j%Ncol == 0:
            ax.set_ylabel(ylabel[k], fontsize=fs)
            k = k + 1               
                             
    plt.savefig(output_dir+'/epoch_{}_id_{}.png'.format(epoch,idx), bbox_inches='tight',dpi=600)
    plt.close(fig)

    print("epoch {}, done printing".format(epoch))