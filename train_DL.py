import argparse
import os
import numpy as np
import itertools
import scipy.io
import pytorch_ssim

from torch.autograd import Variable

import torch.nn as nn
import torch
from data_plot import plot_pred, load_data
from DL_models import Encoder, Decoder, Discriminator
import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument("--current-dir", type=str, default="C:/Users/Administrator/Desktop/DL-para/", help="data directory")
parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument('--n-data', type=int, default=35500, help="number of data samples")
parser.add_argument('--n-train', type=int, default=35000, help='number of training data')
parser.add_argument('--n-test', type=int, default=50, help='number of testing data')
parser.add_argument("--batch-size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lw", type=float, default=0.01, help="adversarial loss weight")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample-interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date = 'experiments/Oct_01'
exp_dir = opt.current_dir + date + "/N{}_Bts{}_Eps{}_lr{}_lw{}".\
    format(opt.n_train, opt.batch_size, opt.n_epochs, opt.lr, opt.lw)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# loss functions
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.MSELoss()
ssim_loss = pytorch_ssim.SSIM(window_size = 11)

nf, d, h, w = 2, 3, 7, 9

# Initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)
discriminator = Discriminator(inchannels=nf)

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()+discriminator._n_parameters()))

encoder.to(device)
decoder.to(device)
discriminator.to(device)
adversarial_loss.to(device)
pixelwise_loss.to(device)
ssim_loss.to(device)

hdf5_dir = opt.current_dir + "data"
dataloader, x_test, n_out_pixels_test= load_data(hdf5_dir, opt)
print("encoder model: {}".format(encoder))
print("decoder model: {}".format(decoder))
print("discriminator model: {}".format(discriminator))

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        
def test(epoch):
    encoder.eval()
    decoder.eval()
    
    # plot original,latent,reconstructions and random realizations
    n_samples = 5
    idx = np.random.choice(x_test.shape[0], n_samples, replace=False)
    for i in range(n_samples):
        real_imgs = x_test[[idx[i]]]
        real_imgs = (torch.FloatTensor(real_imgs)).to(device)
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        samples_gen  = np.squeeze(decoded_imgs.data.cpu().numpy())
        samples_real = np.squeeze(real_imgs.data.cpu().numpy())       
        z = Variable(Tensor(np.random.normal(0, 1, (1, nf, d, h, w))))
        ran_imgs = decoder(z)
        samples_ran  = np.squeeze(ran_imgs.data.cpu().numpy())
        
        print(samples_real.shape, samples_gen.shape, samples_ran.shape)
        samples = np.vstack((samples_real[:10],samples_gen[:10],samples_ran[:10]))
        print(samples.shape)
        plot_pred(samples,epoch,idx[i],output_dir)
                 
 
def pred_test(x_test):
    encoder.eval()
    decoder.eval()
    n_test = x_test.shape[0]
    samples_real = np.zeros_like(np.squeeze(x_test))  
    samples_gen = np.zeros_like(np.squeeze(x_test)) 
    samples_ran = np.zeros_like(np.squeeze(x_test)) 
    samples_z = np.zeros((n_test,nf*d*h*w)) 
    
    # plot original,latent,reconstructions and random realizations  
    for i in range(n_test):
        real_imgs = x_test[[i]]
        real_imgs = (torch.FloatTensor(real_imgs)).to(device)
        with torch.no_grad():
            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
        samples_gen[i]  = np.squeeze(decoded_imgs.data.cpu().numpy())
        samples_real[i] = np.squeeze(real_imgs.data.cpu().numpy())
        samples_z[i] = np.squeeze(encoded_imgs.data.cpu().numpy()).reshape(nf*d*h*w,)
        z = Variable(Tensor(np.random.normal(0, 1, (1, nf, d, h, w))))
        ran_imgs = decoder(z)
        samples_ran[i] = np.squeeze(ran_imgs.data.cpu().numpy())
        
    scipy.io.savemat(output_dir +'/result.mat',  
                     dict(real=samples_real,gen=samples_gen,
                          z=samples_z,ran=samples_ran))
    print(samples_real.shape, samples_gen.shape, samples_z.shape, samples_ran.shape)


def cal_R2_RMSE(epoch):
    encoder.eval()
    decoder.eval()
    n_test = x_test.shape[0]
    mean = np.mean(x_test,axis=0)
    numerator = 0.0
    denominator = 0.0
    SSIM = 0.0
    
    for i in range(n_test):    
        real_imgs = x_test[[i]]
        real_imgs = (torch.FloatTensor(real_imgs)).to(device)
        with torch.no_grad():
            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
        SSIM += (ssim_loss(decoded_imgs[0].cpu(),real_imgs[0].cpu())).item()
        gen = decoded_imgs.data.cpu().numpy()
        real = real_imgs.data.cpu().numpy()        
        
        numerator = numerator + ((real - gen)**2).sum()
        denominator = denominator + ((real - mean)**2).sum()        

    R2 = 1 - numerator/denominator  
    RMSE = np.sqrt(numerator/n_out_pixels_test)
    SSIM = SSIM/n_test   
        
    print("test R2: {} and RMSE: {} at epoch {}".format(R2,RMSE,epoch))
    print("test SSIM: {}".format(SSIM))
    return R2, RMSE, SSIM     


def cal_R2_RMSE_total():
    encoder.eval()
    decoder.eval()   
              
    real_imgs = x_test
    print(real_imgs.shape)
    mean = np.mean(x_test,axis=0)
    numerator = 0.0
    denominator = 0.0    
    real_imgs = (torch.FloatTensor(real_imgs)).to(device)
    with torch.no_grad():
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
    gen = decoded_imgs.data.cpu().numpy()
    real = real_imgs.data.cpu().numpy()
    print(real.shape, gen.shape)
    
    numerator = numerator + ((real - gen)**2).sum()
    denominator = denominator + ((real - mean)**2).sum()    
    R2_total = 1 - numerator/denominator  
    RMSE_total = np.sqrt(numerator/n_out_pixels_test)              
               
         
    R2_test_self = []
    R2_test_self.append(R2_total) 
    np.savetxt(exp_dir + "/R2_test_self.txt", R2_test_self)        
    RMSE_test_self = []
    RMSE_test_self.append(RMSE_total) 
    np.savetxt(exp_dir + "/RMSE_test_self.txt", RMSE_test_self)
    
    return R2_total,RMSE_total      

   
# -----------------------------------
#  Training
# -----------------------------------
RMSE_test, R2_test, SSIM_test = [], [], []

for epoch in range(1,opt.n_epochs+1):
    encoder.train()
    decoder.train()
    discriminator.train()

    for i, (imgs,) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0],1).fill_(1.0), requires_grad=False)
        fake  = Variable(Tensor(imgs.shape[0],1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss_a = adversarial_loss(discriminator(encoded_imgs), valid)
        g_loss_c = pixelwise_loss(decoded_imgs, real_imgs)

        g_loss = opt.lw * g_loss_a + g_loss_c
       
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], nf, d, h, w))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)        
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
    
    if (epoch) % opt.sample_interval == 0:
        test(epoch)

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f /G_A loss: %f/ G_C loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), g_loss_a.item(), g_loss_c.item())
    )

    r2_t, rmse_t, ssim_t = cal_R2_RMSE(epoch)
    R2_test.append(r2_t)
    RMSE_test.append(rmse_t)
    SSIM_test.append(ssim_t)

torch.save(decoder.state_dict(), model_dir + '/DL_decoder_epoch{}.pth'.format(opt.n_epochs))
torch.save(encoder.state_dict(), model_dir + '/DL_encoder_epoch{}.pth'.format(opt.n_epochs))
torch.save(discriminator.state_dict(), model_dir + '/DL_discriminator_epoch{}.pth'.format(opt.n_epochs))
np.savetxt(model_dir + "/R2_test.txt", R2_test)
np.savetxt(model_dir + "/RMSE_test.txt", RMSE_test)
np.savetxt(model_dir + "/SSIM_test.txt", SSIM_test)

# -----------------------------------
#  Testing
# -----------------------------------
# load the pretrained model
print('start predicting...')
encoder.load_state_dict(torch.load(model_dir+'/DL_encoder_epoch{}.pth'.format(opt.n_epochs)),strict=False)
decoder.load_state_dict(torch.load(model_dir+'/DL_decoder_epoch{}.pth'.format(opt.n_epochs)),strict=False)
print("loaded model")
pred_test(x_test)
R2_total,RMSE_total = cal_R2_RMSE_total()
print("R2 of total {} testing samples: {}".format(opt.n_test,R2_total))
print("RMSE of total {} testing samples: {}".format(opt.n_test,RMSE_total))

