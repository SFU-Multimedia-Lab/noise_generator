import numpy as np
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
import scipy.io as sio
#for noise
from PIL import Image
from scipy.stats import ks_2samp
import argparse
import os


def inv_sampling(x,bin_width= 0.00001,test_samples=500):        
        dist_x,mids = dist(x,bin_width)
        cumulative = np.cumsum(dist_x)
        cumulative = cumulative - np.amin(cumulative)
        f = interp1d(cumulative/np.amax(cumulative), mids)
        #plt.hist(x.flatten(),label=['original'])
        #plt.hist(f(np.random.random(test_samples)).flatten(),alpha=0.4, label=['sampled'])
        #plt.legend(loc='upper right')
        #plt.show()
        #f(np.random.random(pnts))
        return f

def dist(x,bin_width = 0.00001 ):
    hist, bin_edges = np.histogram(x, bins=np.linspace(np.amin(x),np.amax(x),int((np.amax(x)-np.amin(x))/bin_width)))
    hist = hist / x.size
    mids = bin_edges[:-1] + np.diff(bin_edges)/2
    return hist,mids

def load_param():
    matdir_b = './dataset_var_samples_q.mat'
    mat_b = sio.loadmat(matdir_b)
    intrcpt_R = mat_b['b_array_R_d']
    intrcpt_G = mat_b['b_array_G_d']
    intrcpt_B = mat_b['b_array_B_d']

    matdir_a_slope_R = './full_dataset_parameter_R.mat'
    matdir_a_slope_G = './full_dataset_parameter_G.mat'
    matdir_a_slope_B = './full_dataset_parameter_B.mat'
    #R parameters
    mat_a_slope_R = sio.loadmat(matdir_a_slope_R)
    a_R = mat_a_slope_R['a_array_R_d']
    m_R = mat_a_slope_R['slope_array_R']
    a_R = a_R[a_R>0]
    #G parameters
    mat_a_slope_G = sio.loadmat(matdir_a_slope_G)
    a_G = mat_a_slope_G['a_array_G_d']
    m_G = mat_a_slope_G['slope_array_G']
    a_G = a_G[a_G>0]
    #B parameters
    mat_a_slope_B = sio.loadmat(matdir_a_slope_B)
    a_B = mat_a_slope_B['a_array_B_d']
    m_B = mat_a_slope_B['slope_array_B']
    a_B = a_B[a_B>0]

    f_intrcpt_R = inv_sampling(intrcpt_R,test_samples=intrcpt_R.size)
    f_intrcpt_G = inv_sampling(intrcpt_G,test_samples=intrcpt_G.size)
    f_intrcpt_B = inv_sampling(intrcpt_B,test_samples=intrcpt_B.size)

    f_m_R = inv_sampling(m_R,test_samples=m_R.size)
    f_m_G = inv_sampling(m_G,test_samples=m_G.size)
    f_m_B = inv_sampling(m_B,test_samples=m_B.size)

    f_a_R = inv_sampling(a_R,test_samples=a_R.size)
    f_a_G = inv_sampling(a_G,test_samples=a_G.size)
    f_a_B = inv_sampling(a_B,test_samples=a_B.size)

    return f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B

def sample_param(f_intercept,f_slope,f_a,n_samples=1):
    
    intercept = f_intercept(np.random.random(n_samples))
    slope     = f_slope(np.random.random(n_samples))
    a         = f_a(np.random.random(n_samples))
    b         = slope*a+intercept
    return a,b

def sample_param_RGB(f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B):
    repeat = True
    while repeat:
        a_R,b_R = sample_param(f_intrcpt_R,f_m_R,f_a_R)
        if b_R>0:
            repeat = False
    
    repeat = True
    while repeat:
        a_G,b_G = sample_param(f_intrcpt_G,f_m_G,f_a_G)
        if b_G>0:
            repeat = False

    repeat = True
    while repeat:
        a_B,b_B = sample_param(f_intrcpt_B,f_m_B,f_a_B)
        if b_B>0:
            repeat = False

    a = np.array([a_R[0],a_G[0],a_B[0]])
    b = np.array([b_R[0],b_G[0],b_B[0]])
    return a,b

def add_noise(img, a_array, b_array):
    #print(img.shape)
    img_dim = img.ndim
    if img_dim==2:
        ch_n = 1.0
        print("Warning: Code is for RGB noise")
    else:
        ch_n = img.shape[2]
    z= np.zeros(img.shape)
    for i in np.arange(0,ch_n):
        y = img[:,:,i]
        a = a_array[i]
        b = b_array[i]
        if a==0:   # no Poissonian component
            z_i=y
        else:      #% Poissonian component
            chi=1./a
            z_i = np.random.poisson(np.maximum(0,chi*y))/chi
 
        z_i=z_i+np.sqrt(np.maximum(0,b))*np.random.normal(loc=0.0, scale=1.0, size=y.shape)  #% Gaussian component
        z[:,:,i] = z_i
    #clipping
    z = np.clip(z, 0.0, 1.0)
    return z
 
def to_ImageFromArray(a):
       return Image.fromarray((a * 255.0).round().clip(0, 255).astype(np.uint8))
       
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument("--img_dir", help="Directory of input image")
    parser.add_argument("--n_obs", default=1, type=int, help="Number of generated noisy image for each image")
    parser.add_argument("--out_dir", help="Directory of output image")
    args = parser.parse_args()

    gt_dir = args.img_dir 
    img_gt = np.asarray(Image.open(gt_dir),dtype=np.float64)/255.0
    n_obs = args.n_obs
    f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B = load_param()

    for i in np.arange(n_obs):
        #sample noise parameters
        a,b = sample_param_RGB(f_intrcpt_R,f_m_R,f_a_R,f_intrcpt_G,f_m_G,f_a_G,f_intrcpt_B,f_m_B,f_a_B)
        #print(a,b)
        #add noise to image and clip
        img_syn_noisy = add_noise(img_gt,a,b) 
        #scale and qunatize
        img_syn_noisy_q = (img_syn_noisy * 255.0).round().clip(0, 255).astype(np.uint8)
        img_name = os.path.basename(gt_dir)
        img_name = " ".join(img_name.split(".")[:-1]) 
        output_name = args.out_dir+"/"+img_name+"_n"+str(i+1)+".png"
        #show/save       
        to_ImageFromArray(img_syn_noisy).save(output_name)   
        #to_ImageFromArray(img_syn_noisy).show()
