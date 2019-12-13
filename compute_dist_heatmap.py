# from models import dist_model
# model = dist_model.DistModel()
from os.path import join
import models
import util.util as util
import matplotlib.pylab as plt
use_gpu = True
fig_outdir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\ImageDiffMetric"
#%%
net_name = 'squeeze'
SpatialDist = models.PerceptualLoss(model='net-lin', net=net_name, colorspace='rgb', spatial=True, use_gpu=True, gpu_ids=[0])
PerceptLoss = models.PerceptualLoss(model='net-lin', net=net_name, colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0])
#%%
imgdir = r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-191212a\backup_12_12_2019_10_47_39"
file0 = "block048_thread000_gen_gen047_001896.jpg"
file1 = "block048_thread000_gen_gen047_001900.jpg"
img0_ = util.load_image(join(imgdir,file0))
img1_ = util.load_image(join(imgdir,file1))
img0 = util.im2tensor(img0_) # RGB image from [-1,1]
if(use_gpu):
    img0 = img0.cuda()
img1 = util.im2tensor(img1_)
if(use_gpu):
    img1 = img1.cuda()
#%
# Compute distance
dist01 = SpatialDist.forward(img0,img1)#.item()
dist_sum = PerceptLoss.forward(img0,img1).item()
# dists.append(dist01)
# print('(%s, %s): %.3f'%(file0,file1,dist01))
# f.writelines('(%s, %s): %.3f'%(file0,file1,dist01))
# %
plt.figure(figsize=[9,3.5])
plt.subplot(131)
plt.imshow(img0_)
plt.subplot(132)
plt.imshow(img1_)
plt.subplot(133)
plt.pcolor(dist01.cpu().detach().squeeze())
plt.axis('image')
plt.gca().invert_yaxis()
plt.title("Dist %.2f"%dist_sum)
plt.savefig(join(fig_outdir,"Diff1212_1896_1900_%s.png" % net_name))
plt.show()