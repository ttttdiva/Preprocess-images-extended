import os

import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from data_loader import create_training_datasets
from model import ISNetDIS, ISNetGTEncoder, U2NET, U2NET_full2, U2NET_lite2, MODNet
from metrics import f1_torch, mae_torch
import pytorch_lightning as pl
import warnings


# warnings.filterwarnings("ignore")


class AnimeSegmentation(pl.LightningModule):

    def __init__(self, net, gt_encoder=None):
        super().__init__()
        assert any([isinstance(net, x) for x in [ISNetDIS, ISNetGTEncoder, U2NET, MODNet]])
        self.net = net
        if gt_encoder is not None:
            assert isinstance(net, ISNetDIS) and isinstance(gt_encoder, ISNetGTEncoder)
            self.gt_encoder = gt_encoder
            for param in self.gt_encoder.parameters():
                param.requires_grad = False
        else:
            self.gt_encoder = None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

    def forward(self, x):
        if isinstance(self.net, ISNetDIS):
            return self.net(x)[0][0].sigmoid()
        if isinstance(self.net, ISNetGTEncoder):
            return self.net(x)[0][0].sigmoid()
        elif isinstance(self.net, U2NET):
            return self.net(x)[0].sigmoid()
        elif isinstance(self.net, MODNet):
            return self.net(x, True)[2]
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetDIS):
            ds, dfs = self.net(images)
            loss_args = [ds, dfs, labels]
        elif isinstance(self.net, ISNetGTEncoder):
            ds = self.net(labels)[0]
            loss_args = [ds, labels]
        elif isinstance(self.net, U2NET):
            ds = self.net(images)
            loss_args = [ds, labels]
        elif isinstance(self.net, MODNet):
            trimaps = batch["trimap"]
            pred_semantic, pred_detail, pred_matte = self.net(images, False)
            loss_args = [pred_semantic, pred_detail, pred_matte, images, trimaps, labels]
        else:
            raise NotImplemented
        if self.gt_encoder is not None:
            fs = self.gt_encoder(labels)[1]
            loss_args.append(fs)

        losses = self.net.compute_loss(loss_args)

        if isinstance(self.net, MODNet):
            semantic_loss, detail_loss, matte_loss = losses
            loss = semantic_loss + detail_loss + matte_loss
            self.log_dict({"train/semantic_loss": semantic_loss, "train/detail_loss": detail_loss,
                           "train/matte_loss": matte_loss})
        else:
            loss0, loss = losses
            self.log_dict({"train/loss": loss, "train/loss_tar": loss0})
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetGTEncoder):
            preds = self.forward(labels)
        else:
            preds = self.forward(images)
        pre, rec, f1, = f1_torch(preds.nan_to_num(nan=0, posinf=1, neginf=0), labels)
        mae = mae_torch(preds, labels)
        pre_m = pre.mean()
        rec_m = rec.mean()
        f1_m = f1.mean()
        mae_m = mae.mean()
        self.log_dict({"val/precision": pre_m, "val/recall": rec_m, "val/f1": f1_m, "val/mae": mae_m})


def get_net(net_name):
    if net_name == "isnet":
        return ISNetDIS()
    elif net_name == "isnet_is":
        return ISNetDIS()
    elif net_name == "isnet_gt":
        return ISNetGTEncoder()
    elif net_name == "u2net":
        return U2NET_full2()
    elif net_name == "u2netl":
        return U2NET_lite2()
    elif net_name == "modnet":
        return MODNet()
    raise NotImplemented


def get_gt_encoder(train_dataloader, val_dataloader, opt):
    print("---start train ground truth encoder---")
    gt_encoder = AnimeSegmentation(get_net("isnet_gt"))
    trainer = Trainer(precision=32 if opt.fp32 else 16, accelerator=opt.accelerator,
                      devices=opt.devices, max_epochs=opt.gt_epoch,
                      benchmark=opt.benchmark, accumulate_grad_batches=opt.acc_step,
                      check_val_every_n_epoch=opt.val_epoch, log_every_n_steps=opt.log_step,
                      )
    trainer.fit(gt_encoder, train_dataloader, val_dataloader)
    return gt_encoder.net


def main(opt):
    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")

    train_dataset, val_dataset = create_training_datasets(opt.data_dir, opt.fg_dir, opt.bg_dir, opt.img_dir,
                                                          opt.mask_dir, opt.fg_ext, opt.bg_ext, opt.img_ext,
                                                          opt.mask_ext, opt.data_split, opt.img_size,
                                                          with_trimap=opt.net == "modnet")

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size_train, shuffle=True,
                                  num_workers=opt.workers_train, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size_val, shuffle=False,
                                num_workers=opt.workers_val, pin_memory=False)
    print("---define model---")
    if opt.resume_ckpt != "":
        anime_seg = AnimeSegmentation.load_from_checkpoint(opt.resume_ckpt, net=get_net(opt.net),
                                                           gt_encoder=get_net(
                                                               "isnet_gt") if opt.net == "isnet_is" else None)
    else:
        net = get_net(opt.net)
        if opt.pretrained_ckpt != "":
            net.load_state_dict(torch.load(opt.pretrained_ckpt))
        if opt.net == "isnet_is":
            gt_encoder = get_gt_encoder(train_dataloader, val_dataloader, opt)
        else:
            gt_encoder = None
        anime_seg = AnimeSegmentation(net, gt_encoder)

    print("---start train---")
    checkpoint_callback = ModelCheckpoint(monitor='val/f1', mode="max", save_top_k=1, save_last=True,
                                          auto_insert_metric_name=False, filename="epoch={epoch},f1={val/f1:.4f}")
    trainer = Trainer(precision=32 if (opt.fp32 or opt.net == "modnet") else 16, accelerator=opt.accelerator,
                      devices=opt.devices, max_epochs=opt.epoch,
                      benchmark=opt.benchmark, accumulate_grad_batches=opt.acc_step,
                      check_val_every_n_epoch=opt.val_epoch, log_every_n_steps=opt.log_step,
                      callbacks=[checkpoint_callback])
    trainer.fit(anime_seg, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--net', type=str, default='isnet_is',
                        choices=["isnet_is", "isnet", "u2net", "u2netl", "modnet"],
                        help='net name')
    parser.add_argument('--pretrained-ckpt', type=str, default='',
                        help='load form pretrained ckpt of net')
    parser.add_argument('--resume-ckpt', type=str, default='',
                        help='resume training from ckpt')

    # dataset args
    parser.add_argument('--data-dir', type=str, default='../../dataset/anime-seg',
                        help='root dir of dataset')
    parser.add_argument('--fg-dir', type=str, default='fg',
                        help='relative dir of foreground')
    parser.add_argument('--bg-dir', type=str, default='bg',
                        help='relative dir of background')
    parser.add_argument('--img-dir', type=str, default='imgs',
                        help='relative dir of images')
    parser.add_argument('--mask-dir', type=str, default='masks',
                        help='relative dir of masks')
    parser.add_argument('--fg-ext', type=str, default='.png',
                        help='extension name of foreground')
    parser.add_argument('--bg-ext', type=str, default='.jpg',
                        help='extension name of background')
    parser.add_argument('--img-ext', type=str, default='.jpg',
                        help='extension name of images')
    parser.add_argument('--mask-ext', type=str, default='.jpg',
                        help='extension name of masks')
    parser.add_argument('--data-split', type=float, default=0.95,
                        help='split rate for training and validation')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='image size for training and validation')

    # training args
    parser.add_argument('--epoch', type=int, default=40,
                        help='epoch num')
    parser.add_argument('--gt-epoch', type=int, default=4,
                        help='epoch for training ground truth encoder when net is isnet_is')
    parser.add_argument('--batch-size-train', type=int, default=2,
                        help='batch size for training')
    parser.add_argument('--batch-size-val', type=int, default=2,
                        help='batch size for val')
    parser.add_argument('--workers-train', type=int, default=4,
                        help='workers num for training dataloader')
    parser.add_argument('--workers-val', type=int, default=4,
                        help='workers num for validation dataloader')
    parser.add_argument('--acc-step', type=int, default=4,
                        help='gradient accumulation step')
    parser.add_argument('--accelerator', type=str, default="gpu",
                        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "auto"],
                        help='accelerator')
    parser.add_argument('--devices', type=int, default=1,
                        help='devices num')
    parser.add_argument('--fp32', action='store_true', default=False,
                        help='disable mix precision (auto disable for modnet)')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='enable cudnn benchmark')
    parser.add_argument('--log-step', type=int, default=2,
                        help='log training loss every n steps')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='valid and save every n epoch')

    opt = parser.parse_args()
    print(opt)

    main(opt)
