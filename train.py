import os
import random
import argparse

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips

from dataset_poem import PoemImageDataset, BertConditioner
from models import weights_init, Discriminator, Generator
from operation import get_dir, InfiniteSamplerWrapper
from diffaug import DiffAugment


# -----------------------------
# Global configs / losses
# -----------------------------
policy = "color,translation"
percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)

lossD_list = []
lossG_list = []


# -----------------------------
# Utilities
# -----------------------------
def zero_center_gp(d_out, x_in):
    """
    Zero-centered gradient penalty: E[||∇x D(x)||^2]
    d_out: D(x) output, any shape but reducible to scalar via sum()
    x_in: input images to D, shape [B,C,H,W], requires_grad=True
    """
    d_sum = d_out.sum()
    grad = torch.autograd.grad(
        outputs=d_sum,
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [B,C,H,W]
    grad = grad.view(grad.size(0), -1)
    return (grad.pow(2).sum(dim=1)).mean()


def crop_image_by_part(image, part):
    """Crop image into one of 4 quadrants."""
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def train_d(netD, data, label="real"):
    """
    Forward-only D step helper.
    Returns:
      - if real: (pred_mean_value, loss_tensor, rec_all, rec_small, rec_part)
      - if fake: loss_tensor
    """
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = netD(data, label, part=part)

        loss_hinge_real = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
        loss_rec_all = percept(rec_all, F.interpolate(data, rec_all.shape[2])).mean()
        loss_rec_small = percept(rec_small, F.interpolate(data, rec_small.shape[2])).mean()
        loss_rec_part = percept(
            rec_part,
            F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]),
        ).mean()

        loss = loss_hinge_real + loss_rec_all + loss_rec_small + loss_rec_part
        return pred.mean().item(), loss, rec_all, rec_small, rec_part

    # label == "fake"
    pred = netD(data, label)
    loss = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
    return loss


class EMA:
    """
    Exponential Moving Average for model parameters.
    - update(): after optimizer step
    - apply_to(): copy EMA weights into model (for eval/saving)
    - restore(): restore original weights back
    """
    def __init__(self, model: nn.Module, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

        if self.device is not None:
            for k in self.shadow:
                self.shadow[k] = self.shadow[k].to(self.device)

    @torch.no_grad()
    def update(self):
        msd = self.model.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self):
        """Apply EMA weights to model (backup current weights first)."""
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow, strict=False)

    def restore(self):
        """Restore model weights from backup."""
        if self.backup is not None:
            self.model.load_state_dict(self.backup, strict=False)
            self.backup = None


# -----------------------------
# Main training
# -----------------------------
def train(args):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size

    ndf = 64
    ngf = 64
    nz = 100
    latent_dim = 100

    nlr = args.lr
    nbeta1 = 0.5

    use_cuda = True
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval

    saved_model_folder, saved_image_folder = get_dir(args)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda and args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    print("Device:", device)

    # 你原来写死的路径（保持不缺失语义）
    # csv_path="/home/ai/code/big/data/poems.csv"
    img_path = "/home/ai//code/data/afhq/stargan-v2/data/train/cat"

    transform = transforms.Compose(
        [
            transforms.Resize((im_size, im_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    dataset = PoemImageDataset(img_dir=img_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=dataloader_workers,
        pin_memory=True,
    )
    dataloader = iter(dataloader)

    # 2. 初始化 BERT（保留注释）
    # bert = BertConditioner(device)

    # 固定样本（用于观察进步；保留变量）
    fixed_sample_batch = next(iter(dataloader))
    fixed_real_imgs = fixed_sample_batch
    fixed_z = torch.randn(batch_size, latent_dim, device=device)

    # Models
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size, use_grassmann=True).apply(weights_init).to(device)
    netD = Discriminator(ndf=ndf, im_size=im_size).apply(weights_init).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    # EMA (新增)
    ema = EMA(netG, decay=0.999, device=None)

    # Resume
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        netG.load_state_dict(ckpt["g"], strict=False)
        netD.load_state_dict(ckpt["d"], strict=False)
        optimizerG.load_state_dict(ckpt["opt_g"])
        optimizerD.load_state_dict(ckpt["opt_d"])
        current_iteration = ckpt.get("iter", 0)
    else:
        current_iteration = 0

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        # -----------------------------
        # 1) Load real & sample fake
        # -----------------------------
        real_image = next(dataloader).to(device)
        bsz = real_image.size(0)

        z = torch.randn(bsz, latent_dim, device=device)
        fake_images = netG(z)  # 你的 G 返回 multi-scale list（保持）

        # DiffAugment
        real_aug = DiffAugment(real_image, policy=policy)
        fake_aug = [DiffAugment(fake, policy=policy) for fake in fake_images]

        # -----------------------------
        # 2) Train Discriminator (Lazy R1/R2)
        # -----------------------------
        optimizerD.zero_grad(set_to_none=True)

        # Real branch: hinge(real) + perceptual recon losses
        err_dr, loss_real, rec_all, rec_small, rec_part = train_d(netD, real_aug, label="real")

        # Fake branch: hinge(fake)
        fake_detached = [fi.detach() for fi in fake_aug]
        loss_fake = train_d(netD, fake_detached, label="fake")

        loss_D = loss_real + loss_fake

        # Lazy R1/R2
        if iteration % 16 == 0:
            # R1 on real
            real_r1 = real_aug.detach().requires_grad_(True)
            d_real_for_r1 = netD(real_r1, label="fake")
            r1 = zero_center_gp(d_real_for_r1, real_r1)

            # R2 on fake (highest-res fake only)
            fake0 = fake_aug[0].detach().requires_grad_(True)
            d_fake_for_r2 = netD(fake0, label="fake")
            r2 = zero_center_gp(d_fake_for_r2, fake0)

            loss_D = loss_D + 0.1 * r1 + 0.1 * r2

        loss_D.backward()
        optimizerD.step()

        # -----------------------------
        # 3) Train Generator
        # -----------------------------
        optimizerG.zero_grad(set_to_none=True)

        pred_g = netD(fake_aug, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        # EMA update (新增)
        ema.update()

        # -----------------------------
        # 4) Logging
        # -----------------------------
        lossD_list.append(err_dr)            # D(real) 的均值输出（保持你原记录方式）
        lossG_list.append(-err_g.item())     # 你原来记录 -err_g

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
            plt.close()
            print("Loss curve saved to:", loss_fig_path)

        if iteration % 100 == 0:
            print(f"GAN: loss d: {err_dr:.5f}    loss g: {-err_g.item():.5f}")

        # -----------------------------
        # 5) Save images (use EMA for prettier samples)
        # -----------------------------
        if iteration % (save_interval * 10) == 0:
            with torch.no_grad():
                # 用 EMA 权重生成固定样本（新增：更稳定）
                ema.apply_to()
                raw_fixed_fake = netG(z)[0]  # 保持你原来用当前 z 的行为
                ema.restore()

                fixed_fake_imgs = raw_fixed_fake.view(bsz, 3, im_size, im_size)
                img_grid = (fixed_fake_imgs + 1.0) / 2.0  # denorm to [0,1]
                save_image(img_grid, f"{saved_image_folder}/iter_{iteration}.jpg", nrow=4)

                # 重建图（保持）
                save_image(
                    torch.cat([rec_all, rec_small, rec_part]).add(1).mul(0.5),
                    f"{saved_image_folder}/rec_{iteration}.jpg",
                )

        # -----------------------------
        # 6) Save checkpoints
        # -----------------------------
        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            torch.save(
                {
                    "iter": iteration,
                    "g": netG.state_dict(),
                    "g_ema": ema.shadow,
                    "d": netD.state_dict(),
                    "opt_g": optimizerG.state_dict(),
                    "opt_d": optimizerD.state_dict(),
                },
                os.path.join(saved_model_folder, f"all_{iteration}.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="region gan")

    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument("--output_path", type=str, default="./", help="Output path for the train results")
    parser.add_argument("--cuda", type=int, default=0, help="index of gpu to use")
    parser.add_argument("--name", type=str, default="Base", help="experiment name")
    parser.add_argument("--iter", type=int, default=2000000, help="number of iterations")
    parser.add_argument("--start_iter", type=int, default=0, help="the iteration to start training")
    parser.add_argument("--batch_size", type=int, default=32, help="mini batch number of images")
    parser.add_argument("--im_size", type=int, default=256, help="image resolution")
    parser.add_argument("--ckpt", type=str, default="./all_165000.pth", help="checkpoint weight path if have one")
    parser.add_argument("--workers", type=int, default=2, help="number of workers for dataloader")
    parser.add_argument("--save_interval", type=int, default=100, help="number of iterations to save model")
    parser.add_argument("--lr", type=float, default=0.0002, help="learn")

    args = parser.parse_args()
    print(args)

    train(args)
