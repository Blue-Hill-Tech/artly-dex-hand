"""Render the supplied D21 URDF geometry offline; no hardware or vendor examples."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=Path, required=True, help='Directory containing supplied D21 URDFs and mesh folders')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
client = p.connect(p.DIRECT)
for side, x in [('Left', -.115), ('Right', .115)]:
    p.loadURDF(str(args.models / f'D21_{side}.urdf'), basePosition=[x, 0, .05 if side == 'Left' else 0], useFixedBase=True, physicsClientId=client)
view=p.computeViewMatrix(cameraEyePosition=[0,-.65,.17],cameraTargetPosition=[0,-.015,.082],cameraUpVector=[0,0,1])
projection=p.computeProjectionMatrixFOV(fov=29,aspect=1040/630,nearVal=.01,farVal=3)
_,_,rgba,depth,seg=p.getCameraImage(2080,1260,viewMatrix=view,projectionMatrix=projection,lightDirection=[-2,-3,4],lightColor=[1,1,1],lightAmbientCoeff=.45,lightDiffuseCoeff=.58,lightSpecularCoeff=.16,shadow=1,renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,physicsClientId=client)
p.disconnect(client)
rgba=np.array(rgba,dtype=np.uint8).reshape(1260,2080,4)
mask=np.asarray(seg).reshape(1260,2080)!=-1
rgba[:,:,3]=mask.astype(np.uint8)*255
models=Image.fromarray(rgba).resize((1040,630),Image.Resampling.LANCZOS)
canvas=Image.new('RGB',(1600,700),(246,248,249))
d=ImageDraw.Draw(canvas)
regular='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
font=lambda size,strong=False:ImageFont.truetype(bold if strong else regular,size)
d.text((76,110),'ARTLY',font=font(23,True),fill=(51,94,93))
d.text((72,170),'Dex Hand',font=font(62,True),fill=(24,40,50))
d.text((76,264),'D21',font=font(32,True),fill=(51,94,93))
d.line((76,336,398,336),fill=(212,222,225),width=2)
d.text((76,374),'Engineering resources',font=font(22),fill=(63,80,90))
d.text((76,416),'SDKs · CAD · Simulation',font=font(20),fill=(91,107,115))
canvas.paste(models,(500,10),models)
d.text((76,623),'Blue Hill Tech, Inc. / Artly',font=font(16),fill=(91,107,115))
d.text((1500,623),'Supplier CAD model renders · D21',anchor='ra',font=font(16),fill=(91,107,115))
args.output.parent.mkdir(parents=True,exist_ok=True)
canvas.save(args.output)
print(args.output)
