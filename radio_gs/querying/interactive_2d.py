"""Reference-only SAM prompt parsing; output enters the field's image query path."""
import numpy as np
from PIL import Image


def validate_2d_prompt(width,height,*,points=None,labels=None,box=None):
    if width<1 or height<1:raise ValueError('invalid image dimensions')
    if box is not None:
        if points is not None or labels is not None:raise ValueError('box and points require an explicit composition policy; decode independently')
        b=np.asarray(box,dtype=np.float32)
        if b.shape!=(4,) or not np.isfinite(b).all() or not(0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height):raise ValueError('invalid pixel XYXY box')
        return {'box':b}
    p=np.asarray(points,dtype=np.float32);l=np.asarray(labels)
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or l.shape!=(len(p),) or not np.isfinite(p).all() or not np.isin(l,[0,1]).all() or not (l==1).any():raise ValueError('finite signed XY points with a positive point required')
    if ((p[:,0]<0)|(p[:,0]>width-1)|(p[:,1]<0)|(p[:,1]>height-1)).any():raise ValueError('point outside image pixel centres')
    for i in range(len(p)):
        if ((np.all(p==p[i],axis=1))&(l!=l[i])).any():raise ValueError('contradictory signed point')
    return {'point_coords':p,'point_labels':l.astype(np.int32)}


def scribble_points(positive,negative,maximum_per_sign=8):
    positive=np.asarray(positive);negative=np.asarray(negative)
    if positive.dtype!=bool or negative.dtype!=bool or positive.ndim!=2 or positive.shape!=negative.shape or (positive&negative).any() or not positive.any() or maximum_per_sign<1:raise ValueError('invalid or contradictory signed scribbles')
    points=[];labels=[]
    for mask,sign in [(positive,1),(negative,0)]:
        xy=np.argwhere(mask)[:,::-1].astype(np.float32)
        if not len(xy):continue
        chosen=[0];distance=((xy-xy[0])**2).sum(1)
        for _ in range(min(maximum_per_sign,len(xy))-1):
            j=int(distance.argmax());chosen.append(j);distance=np.minimum(distance,((xy-xy[j])**2).sum(1))
        points.extend(xy[chosen]);labels.extend([sign]*len(chosen))
    return np.asarray(points,dtype=np.float32),np.asarray(labels,dtype=np.int32)


def select_prompt_mask(masks,quality,*,positive=None,negative=None):
    masks=np.asarray(masks);quality=np.asarray(quality).reshape(-1)
    if masks.ndim!=3 or masks.dtype!=bool or len(masks)!=len(quality) or not len(masks) or not np.isfinite(quality).all():raise ValueError('invalid SAM candidate masks')
    signs=[]
    for evidence,label in [(positive,True),(negative,False)]:
        if evidence is None:continue
        evidence=np.asarray(evidence)
        if evidence.dtype!=bool or evidence.shape!=masks.shape[1:]:raise ValueError('prompt evidence raster differs')
        if evidence.any():signs.append((evidence,label))
    if len(signs)==2 and (signs[0][0]&signs[1][0]).any():raise ValueError('contradictory prompt evidence')
    rows=[]
    for i,m in enumerate(masks):
        agreement=[float((m[e]==sign).mean()) for e,sign in signs]
        rows.append({'index':i,'signed_agreement':float(np.mean(agreement)) if agreement else None,'all_signed_pixels_satisfied':all(x==1 for x in agreement) if agreement else None,'quality':float(quality[i]),'positive_pixels':int(m.sum())})
    best=max(range(len(rows)),key=lambda i:((rows[i]['signed_agreement'] if signs else 0),rows[i]['quality'],-i))
    return masks[best].copy(),{'selection':'full authorized signed evidence, then SAM quality; no target mask','selected':rows[best],'candidates':rows}


def masked_query_crop(image,mask,padding=3):
    mask=np.asarray(mask)
    if mask.dtype!=bool or mask.shape!=(image.height,image.width) or padding<0 or not mask.any():raise ValueError('nonempty reference mask in RGB coordinates required')
    rgb=np.asarray(image.convert('RGB')).copy();rgb[~mask]=127;y,x=np.where(mask);box=(max(0,int(x.min())-padding),max(0,int(y.min())-padding),min(image.width,int(x.max())+padding+1),min(image.height,int(y.max())+padding+1))
    return Image.fromarray(rgb).crop(box),box
