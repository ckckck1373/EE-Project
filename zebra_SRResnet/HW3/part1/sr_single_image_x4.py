# Proximal
import sys

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import cvxpy as cvx
import numpy as np
from scipy import signal

from PIL import Image
import cv2

import scipy.misc
import time

from calc_psnr import  *

############################################################

# set source image
img_name = 'zebra_test'
img_dir = 'image'
gt_img_filename = './ref/%s_sr_single.png' % (img_name)

# set parameters

gaussian_std = 1.5
lamb = 1e-2

#======================================================

lr_prefix = 'LR'
img_filename = './%s/%s_%s.png' % (img_dir, lr_prefix, img_name)

# Load image
img = Image.open(img_filename)  # opens the file using Pillow - it's not an array yet
np_img = np.asfortranarray(im2nparray(img))



# You should write gaussian filter by yourself
#=====================================================
# Construct Gaussian filter
def get_kernel():



    
    return gau_kernel
    
gau_kernel = get_kernel()

#=====================================================


gau_rgb = np.zeros((gau_kernel.shape[0],gau_kernel.shape[1],3))
gau_rgb[:,:,0] = gau_kernel
gau_rgb[:,:,1] = gau_kernel
gau_rgb[:,:,2] = gau_kernel

# Now test the solver with some sparse gradient deconvolution
eps_abs_rel = 1e-3
test_solver = 'pc'
max_iters = 1000

tstart = time.time()
 
#rgb channels
b = np_img
b_upscale = cv2.resize(b,(0,0),fx=4,fy=4, interpolation = cv2.INTER_CUBIC)

x = Variable((b.shape[0]*4,b.shape[1]*4,b.shape[2]))

prob = Problem(norm1( subsample(conv(gau_rgb,x, dims=2),(4,4,1)) - b ) + lamb * group_norm1( grad(x, dims = 2), [3] )  ) # formulate problem

result = prob.solve(verbose=True, solver = test_solver, x0 = b1_upscale, eps_abs = eps_abs_rel, \ 
            eps_rel=eps_abs_rel,max_iters=max_iters) # solve problem
x = x.value

t_int = time.time() - tstart
print( "Elapsed time: %f seconds.\n" %t_int )

#output result

gt_img = Image.open(gt_img_filename)
np_gt_img = np.asfortranarray(im2nparray(gt_img))

psnr = calc_psnr(np_gt_img,x)
print 'PSNR: %f'%(psnr)

scipy.misc.toimage(x, cmin=0.0, cmax=1.0).save('result/%s_sr_single_image_x4_std_%.4f_la%.4f_psnr%.2f.png' %(img_name,gaussian_std,lamb,psnr))
