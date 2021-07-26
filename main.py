import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import LearningRateMonitor
import pl_bolts
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse  
from resnet import resnet18 
import os
from knn_predictor import BenchmarkModule
from data import simsiam_cifar10_loader, cifar10_loader
import copy

parser = argparse.ArgumentParser(description='SimSiam Training')

parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size')
parser.add_argument('--learning-rate', default=0.06, type=float,help='base learning rate')

parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--model-path',default=None)

parser.add_argument('--knn-k', default=200, type=int)
parser.add_argument('--knn-t', default=0.1, type=int)

parser.add_argument('--in-dim', default=512, type=int)
parser.add_argument('--h-dim', default=1024, type=int)
parser.add_argument('--out-dim', default=256, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--taw', default=0.996, type=float)

parser.add_argument('--save-dir', default = '/gdrive/MyDrive/simsiam/', type=str)
parser.add_argument('--exp-name', default = 'resnet_2048_512', type=str)

args=parser.parse_args()

logger = TensorBoardLogger(os.path.join(args.save_dir,args.exp_name), name="tensorboard")

class SimSiam(BenchmarkModule):

    def __init__(self, dataloader_kNN, gpus=1, classes=10, knn_k=args.knn_k, knn_t=args.knn_t, in_dim=args.in_dim,h_dim=args.h_dim,out_dim=args.out_dim,taw=args.taw, model_path=None):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        self.online_resnet = resnet18()   #torchvision.models.resnet18(pretrained=False)
        self.target_resnet = resnet18()   #copy.copy(self.online_resnet)  
        self.taw=taw 
        self.step=0
        self.total_steps=len(dataloader_kNN) * args.epochs
        self.mse_loss=torch.nn.MSELoss()

        self.online_resnet.fc = nn.Sequential(nn.Linear(in_dim, h_dim, bias=False),  #projection [512x1024]
                                        nn.BatchNorm1d(h_dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(h_dim, out_dim, bias=False),   #[1024x512]
                                       ) 
        self.target_resnet.fc = nn.Sequential(nn.Linear(in_dim, h_dim, bias=False),  #projection [512x1024]
                                        nn.BatchNorm1d(h_dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(h_dim, out_dim, bias=False),   #[1024x512]
                                       )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(out_dim, h_dim, bias=False),
                                        nn.BatchNorm1d(h_dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(h_dim, out_dim)) 

        if model_path and os.path.exists(model_path):
          print('loading the model.....')
          ckpt=torch.load(model_path)
          self.online_resnet.load_state_dict(ckpt['online_resnet'])
          self.predictor.load_state_dict(ckpt['predictor'])



    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.online_resnet(x)
        return embedding

    def D(self, p, z): # negative cosine similarity
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize each emb vector
        z = F.normalize(z, dim=1) # l2-normalize each emb vector
        return -(p*z).sum(dim=1).mean()


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06, momentum=0.9, weight_decay=5e-4)
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=10, max_epochs=800, warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)
        return [optimizer], [scheduler]

if __name__ == '__main__':

    log_dir = os.path.join(args.save_dir,args.exp_name,'tensorboard')
    save_dir = os.path.join(args.save_dir,args.exp_name,'model.ckpt')

    seed_everything(args.seed)

    early_stop_callback = EarlyStopping(monitor='kNN_accuracy', min_delta=0.00, patience=5, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint( monitor='kNN_accuracy', dirpath= os.path.join(args.save_dir,args.exp_name,'checkpoints'), filename='online_resnet-epoch-{epoch}-acc-{kNN_accuracy:.2f}', mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    knn_train_loader, knn_val_loader=cifar10_loader(args.batch_size)
    simsiam = SimSiam(knn_train_loader,model_path=args.model_path)
    trainer = Trainer( gpus=1, max_epochs=args.epochs, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,
                      progress_bar_refresh_rate=1,callbacks=[lr_monitor],logger=logger)

    train_loader, _ = simsiam_cifar10_loader(args.batch_size)
    
    trainer.fit(simsiam, train_loader, knn_val_loader)
    save({'online_resnet':simsiam.online_resnet.state_dict(),'predictor':simsiam.predictor.state_dict()},  save_dir)
