import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torchvision.utils import save_image
import argparse
import os
import random
from tqdm import tqdm
from dataset_poem import PoemImageDataset, BertConditioner 
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
import lpips
import matplotlib.pyplot as plt
policy = 'color,translation'
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
lossD_list = []
lossG_list = []
def zero_center_gp(d_out, x_in):
    """
    d_out: D(x) 的输出，形状通常是 [B] 或 [2B] 或 [B, ...] 都行（你这里是 torch.cat 后的一维向量）:contentReference[oaicite:1]{index=1}
    x_in: 输入到 D 的真实图像张量，shape [B,C,H,W]，requires_grad=True
    """
    # 保证是标量输出做 grad
    d_sum = d_out.sum()
    grad = torch.autograd.grad(
        outputs=d_sum,
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [B,C,H,W]
    grad = grad.view(grad.size(0), -1)
    return (grad.pow(2).sum(dim=1)).mean()  # E[||∇x D(x)||^2]

    
def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Return (pred_mean, loss_tensor, optional rec imgs...) WITHOUT backward."""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)

        loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
               percept(rec_all,  F.interpolate(data, rec_all.shape[2])).mean() + \
               percept(rec_small, F.interpolate(data, rec_small.shape[2])).mean() + \
               percept(rec_part,  F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).mean()

        return pred.mean().item(), loss, rec_all, rec_small, rec_part

    else:
        pred = net(data, label)
        loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        return loss
    
def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 100
    nlr = args.lr
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    latent_dim = 100
    #csv_path="/home/ai/code/big/data/poems.csv"
    img_path="/home/ai//code/data/afhq/stargan-v2/data/train/cat"
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder, saved_image_folder = get_dir(args)

    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")
    print(device)
    transform = transforms.Compose([
        transforms.Resize((args.im_size, args.im_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = PoemImageDataset(img_dir=img_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=args.workers,
        pin_memory=True
    )
    dataloader = iter(dataloader)
    
    # 2. 初始化 BERT
    #bert = BertConditioner(device)
    
    # 从数据集中拿出一组固定的诗句和噪声，用于观察每个阶段的进步
    fixed_sample_batch = next(iter(dataloader))
    fixed_real_imgs = fixed_sample_batch
    
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size,use_grassmann=True)
    netG.apply(weights_init)
    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)
    netG.to(device)
    netD.to(device)
    avg_param_G = copy_G_params(netG)
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        
        netG.load_state_dict(ckpt["g"], strict=False)
        netD.load_state_dict(ckpt["d"], strict=False)
        # optimizerG.load_state_dict(ckpt["opt_g"])
        # optimizerD.load_state_dict(ckpt["opt_d"])
        
        current_iteration = ckpt.get("iter", 0)
    else:
        current_iteration = 0
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)  # poems 是原始文本列表
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        z = torch.randn(current_batch_size, latent_dim, device=device)
        fake_images = netG(z)
        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator (with Lazy R1)
        optimizerD.zero_grad(set_to_none=True)
        # 1) 真实分支：你原来的“重建 + hinge real”loss（label="real" 走重建）
        err_dr, loss_real, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        # 2) 假分支：你原来的 hinge fake loss（输入是 list）
        fake_detached = [fi.detach() for fi in fake_images]
        loss_fake = train_d(netD, fake_detached, label="fake")
        loss_D = loss_real + loss_fake
        # R1\R2 的 损失
        if (iteration % 16) == 0:
            # -------- R1 on real --------
            real_image = real_image.detach().requires_grad_(True)
            d_real_for_r1 = netD(real_image, label="fake")
            r1 = zero_center_gp(d_real_for_r1, real_image)
        
            # -------- R2 on fake (use highest-res fake only) --------
            fake0 = fake_images[0].detach().requires_grad_(True)  # leaf tensor
            d_fake_for_r2 = netD(fake0, label="fake")           # 注意：D 期待 list，这里传 [fake0]
            r2 = zero_center_gp(d_fake_for_r2, fake0)

            loss_D = loss_D + 0.1 * r1 + 0.1 * r2
        
        loss_D.backward()
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()
        
        lossD_list.append(err_dr)  # 记录 D loss 数值
        lossG_list.append(-err_g.item())  # 记录 G loss 数值
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 200 == 0:
            steps = list(range(len(lossD_list)))
            plt.figure(figsize=(8, 5))
            plt.plot(steps, lossD_list, label="D(real)")
            plt.plot(steps, lossG_list, label="D(fake)")
            plt.legend()
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("GAN Training Loss")
            loss_fig_path = os.path.join(saved_image_folder, "loss_curve.png")
            plt.savefig(loss_fig_path)
            print("Loss curve saved to:", loss_fig_path)
            
        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))
          
        if iteration % (save_interval*10) == 0:
            with torch.no_grad():
                raw_fixed_fake = netG(z)[0]
                fixed_fake_imgs = raw_fixed_fake.view(current_batch_size, 3, im_size, im_size)
                # 反归一化到 [0,1]
                img_grid = (fixed_fake_imgs + 1.0) / 2.0 
                save_image(img_grid, f"{saved_image_folder}/iter_{iteration}.jpg", nrow=4)
                save_image( torch.cat([ 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
        
        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            torch.save({
                "iter": iteration,
                "g": netG.state_dict(),
                "d": netD.state_dict(),
                "opt_g": optimizerG.state_dict(),
                "opt_d": optimizerD.state_dict(),
            }, os.path.join(saved_model_folder, f"all_{iteration}.pth"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='Base', help='experiment name')
    parser.add_argument('--iter', type=int, default=2000000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=32, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='./all_165000.pth', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--save_interval', type=int, default=100, help='number of iterations to save model')
    parser.add_argument('--lr', type=float, default=0.0002, help='learn')
   
    args = parser.parse_args()
    print(args)

    train(args)
