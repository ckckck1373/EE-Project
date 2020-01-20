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
gt_img_filename = './ref/%s_sr_four.png' % (img_name)
# set parameters
gaussian_std = 1.5
lamb = 1e-2

#=====================================================

# Construct Gaussian filter
# and remember that
# gau_rgb[:,:,0] = gau_kernel
# gau_rgb[:,:,1] = gau_kernel
# gau_rgb[:,:,2] = gau_kernel

def get_kernel(mv_x, mv_y):
    upsample_factor = 4






    return gau_rgb
#======================================================

lr_prefix = 'LR'
img_filename = './%s/%s_%s' % (img_dir, lr_prefix, img_name)


# Load image
mv_x = 0.0
mv_y = 0.0
img1 = Image.open('%s_mvx%.2f_mvy%.2f.png'%(img_filename,mv_x,mv_y))  # opens the file using Pillow - it's not an array yet
np_img1 = np.asfortranarray(im2nparray(img1))
gau_rgb1 = get_kernel(mv_x, mv_y)

mv_x = -0.5
mv_y = 0.0
img2 = Image.open('%s_mvx%.2f_mvy%.2f.png'%(img_filename,mv_x,mv_y))  # opens the file using Pillow - it's not an array yet
np_img2 = np.asfortranarray(im2nparray(img2))
gau_rgb2 = get_kernel(mv_x, mv_y)

mv_x = 0.0
mv_y = -0.5
img3 = Image.open('%s_mvx%.2f_mvy%.2f.png'%(img_filename,mv_x,mv_y))  # opens the file using Pillow - it's not an array yet
np_img3 = np.asfortranarray(im2nparray(img3))
gau_rgb3 = get_kernel(mv_x, mv_y)

mv_x = -0.5
mv_y = -0.5
img4 = Image.open('%s_mvx%.2f_mvy%.2f.png'%(img_filename,mv_x,mv_y))  # opens the file using Pillow - it's not an array yet
np_img4 = np.asfortranarray(im2nparray(img4))
gau_rgb4 = get_kernel(mv_x, mv_y)

# Now test the solver with some sparse gradient deconvolution
eps_abs_rel = 1e-3
test_solver = 'pc'
max_iters = 1000

tstart = time.time()
 
#rgb channels
b1_upscale = cv2.resize(np_img1,(0,0),fx=4,fy=4, interpolation = cv2.INTER_CUBIC)
x = Variable((np_img1.shape[0]*4,np_img1.shape[1]*4,np_img1.shape[2]))

# Do the problem definition by yourself
#=====================================================






#=====================================================

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

scipy.misc.toimage(x, cmin=0.0, cmax=1.0).save('result/%s_sr_four_image_x4_std_%.4f_la%.4f_psnr%.2f.png' %(img_name,gaussian_std,lamb,psnr))
