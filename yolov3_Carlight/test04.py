import cfg
import PIL.Image as pimg
import torch
import os
import PIL.ImageDraw as draw
import utils as ut
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm_t=torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
IMG_BASE_DIR = r"test_img"
path_out = r"./img_out"

def cmt(x,num,p):
    x=x.permute(0,3,2,1)#nwhc
    x=x[0].view(num,num,3,5+cfg.CLASS_NUM)
    c=torch.nonzero(torch.gt(x[:,:,:,0],p)).float()
    cx = x[:,:,:,1][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    cy = x[:,:,:,2][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    w = x[:,:,:,3][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    h = x[:,:,:,4][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
    if c.size(0)==0:
        s=torch.tensor([]).view(c.size(0),0).to(device)
    else:
        s=torch.tensor([]).view(c.size(0),-1).to(device)
    for i in range(cfg.CLASS_NUM):
        ss = x[:,:,:,5+i][torch.gt(x[:,:,:,0],p)].view(-1,1).float()
        s=torch.cat((s,ss),1)
    if s.size(0)==0:
        return torch.tensor([]).view(-1,5).to(device)
    a,s=torch.max(s,dim=1)
    s=s.view(-1,1).float()
    tk=torch.tensor(cfg.ANCHORS_GROUP[num])[c[:,2].cpu().numpy().tolist()].to(device).float()*torch.exp(torch.cat((w,h),1))
    out=torch.cat((s,c[:,0:1]*cfg.IMG_WIDTH/num+num*cx-tk[:,:1]/2,
                   c[:,1:2]*cfg.IMG_HEIGHT/num+num*cy-tk[:,1:]/2,
                   c[:,0:1]*cfg.IMG_WIDTH/num+num*cx+tk[:,:1]/2,
                   c[:,1:2]*cfg.IMG_HEIGHT/num+cy*num+tk[:,1:]/2),1)
    return out
        
def test():
    net = torch.load(r'net1006_num15_sigmoid.pth').to(device)
    net.eval()
    color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#111111', '#ffffff']
    for i in os.listdir(IMG_BASE_DIR):
        img=pimg.open(os.path.join(IMG_BASE_DIR,i))
        im=torchvision.transforms.ToTensor()(img)
        im=norm_t(im).view(1,3,416,416).to(device)
        out_13,out_26,out_52=net(im)
        out_all=torch.cat((cmt(out_13,13,0.5),cmt(out_26,26,0.5),cmt(out_52,52,0.5)),0).int().cpu().detach().numpy().tolist()
        img_draw = draw.ImageDraw(img)
        for j in out_all:
            img_draw.rectangle(j[1:],fill=None,outline=color[j[0]-1])
        img.show()
        # img.save('{}/{}.png'.format(path_out,str(i)+str(a)))

if __name__ == '__main__':
    test()
    
    
    
    




































