import wandb
import os
import time
from . import util
from pathlib import Path
class Logger_wb():
    def __init__(self, opt):
        wandb.init(
            # Set the project where this run will be logged
            project="style_transfer_cyclegan", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{opt.model}_{opt.netG}_{opt.netD}_{opt.dataroot}_{opt.preprocess}_{opt.load_size}_{opt.crop_size}_{opt.no_flip}", 
            # Track hyperparameters and run metadata
            config={
            "model":opt.model,
            "netG":opt.netG,
            "netD":opt.netD,
            "dataroot":opt.dataroot,
            "dataset_mode":opt.dataset_mode,
            "serial_batches":opt.serial_batches,
            "input_nc": opt.input_nc,
            "n_layers_D": opt.n_layers_D,

            
            "preprocess": opt.preprocess,

            "load_size": opt.load_size,
            "crop_size": opt.crop_size,

            "batch_size": opt.batch_size ,
            "n_epochs": opt.n_epochs,
            "n_epochs_decay": opt.n_epochs_decay,
            
            "lr": opt.lr,
            "lr_decay_iters": opt.lr_decay_iters ,
            "lr_policy": opt.lr_policy ,
            "beta1": opt.beta1,
            
        })

        Path(os.path.join(opt.checkpoints_dir, opt.name)).mkdir(parents=True, exist_ok=True)
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def plot_current_losses(self, epoch, counter_ratio, losses):
            """display the current losses on visdom display: dictionary of error labels and values

            Parameters:
                epoch (int)           -- current epoch
                counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
                losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
            """
            if len(losses) == 0:
                return
            train_loss = {}
            train_loss['train'] = losses

            meta_losses = {}
            meta_losses["epoch"] = epoch
            meta_losses["counter_ratio"] = counter_ratio

            wandb.log({"train": dict(losses), "meta": meta_losses})
    
    def plot_current_eval(self, eval):
            """display the current eval: dictionary of error labels and values

            Parameters:
                eval   -- evaluations metrices stored in the format of (name, float) pairs
            """
            if len(eval) == 0:
                return

            wandb.log({"eval": eval})
            
    def display_current_results(self, visuals, epoch= None, save_result= None):          

        wandb.log({"images": [wandb.Image(util.tensor2im(image), caption=label)  for label, image in visuals.items()]})

    def finish(self):
        wandb.finish()

        # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message