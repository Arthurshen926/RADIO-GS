import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
root=Path('output/lerf_support_decomposition_next19')
frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label/waldo_kitchen'),frames=['frame_00066','frame_00154'])
fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for row,(frame,query) in enumerate([('frame_00066','plastic ladle'),('frame_00154','plate')]):
 obj=next(x for x in frames[frame] if x.query==query);mask=obj.mask;y,x=np.where(mask);ys=slice(max(0,y.min()-20),min(mask.shape[0],y.max()+21));xs=slice(max(0,x.min()-20),min(mask.shape[1],x.max()+21));data=torch.load(root/'waldo_kitchen'/(frame+'.pt'),weights_only=False)
 for col,(key,title) in enumerate([('alpha','Geometry opacity'),('observed_fraction','Native semantic support'),('valid','Deployed validity mask')]):
  ax=axes[row,col];ax.imshow(data[key].numpy()[ys,xs],vmin=0,vmax=1,cmap='viridis');ax.contour(mask[ys,xs],levels=[.5],colors=['red'],linewidths=.8);ax.set_title(query+' — '+title,fontsize=10);ax.axis('off')
fig.suptitle('Opaque projection exists, but semantic evidence is missing\nRed outlines: diagnostic GT; opacity does not verify target-object geometry',fontsize=12)
fig.savefig(root/'support_decomposition.png',dpi=160);plt.close(fig)
