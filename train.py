"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.logger_wb import Logger_wb
import warnings
from util.fid_score import InceptionV3, calculate_fretchet
import torch
import torchvision.transforms as transforms
import random
import math
def train(model, dataset_training, dataset_eval, inception_model, opt, logger):
    
    total_iters = 0                # the total number of training iterations
    dataset_size = len(dataset_training)    # get the number of images in the dataset.
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset_training):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                eval(model, dataset_eval, inception_model, opt, logger)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                if logger is not None: logger.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    if logger is not None: logger.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

def eval(model, dataset_eval, inception_model, opt, logger):
    opt.phase = "eval"
    model.eval()
    total_visuals = []
    for i, data in enumerate(dataset_eval):       
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        total_visuals.append(visuals)
    model.train()
    opt.phase = "train"

    # calculate MSE for real_b and fake_b (Note data set is paired)
    eval_log = {}
    if opt.input_nc == 1:
        normalizer = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    else: 
        normalizer = transforms.Compose([transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))])
    if opt.input_nc == 1:
        images_real = normalizer(torch.squeeze(torch.stack( [visuals["real_B"].repeat(1,3,1,1) for visuals in total_visuals])))
        images_fake = normalizer(torch.squeeze(torch.stack( [visuals['fake_B'].repeat(1,3,1,1) for visuals in total_visuals])))
    else:
        images_real = normalizer(torch.squeeze(torch.stack( [visuals["real_B"] for visuals in total_visuals])))
        images_fake = normalizer(torch.squeeze(torch.stack( [visuals['fake_B'] for visuals in total_visuals]))) 


    eval_log['FID'] = calculate_fretchet(images_real, images_fake, inception_model)

    
    if logger is not None:
        logger.plot_current_eval(eval_log)
        logger.display_current_results(random.choice(total_visuals))

if __name__ == '__main__':

    log = True
    opt = TrainOptions().parse()   # get training options
    
    if opt.job_id != -1:
        # TODO change the test image to be equal to crop size instead of load_size
        opt.load_size = random.choices([512, 256])[0]
        opt.crop_size = random.choices([128, 64])[0]
        opt.serial_batches = False
        opt.preprocess = "resize_and_crop"
        opt.netG = random.choices(["resnet_9blocks", "resnet_6blocks", "unet_128", "unet_256"])[0]
        if opt.netG in ["resnet_9blocks", "resnet_6blocks"]:
            opt.n_downsampling = random.choices([2,3,4])[0]
        elif opt.netG == "unet_128":
            opt.crop_size = 128
            crop_log = int(math.log2(opt.crop_size))
            opt.n_downsampling = random.choices([crop_log, crop_log-1, crop_log-2])[0]
        elif opt.netG == "unet_256":
            opt.crop_size = 256
            crop_log = int(math.log2(opt.crop_size))
            opt.n_downsampling = random.choices([crop_log, crop_log-1, crop_log-2])[0]

        #opt.netD = random.choices(['n_layers'])[0]
        #opt.n_layers_D = random.choices([4])[0]
        opt.batch_size = random.choices([1])[0]
        #opt.init_gain = random.choices([0.002, 0.01,0.02])[0]
        # opt.ngf = random.choices([64])[0]
        # opt.ndf = random.choices([32])[0]
        opt.lambda_identity = random.choices([0.25,0.5,1])[0]
        opt.lambda_A = random.choices([5,10,15])[0]
        opt.lambda_B = random.choices([5,10,15])[0]
        

    
    
    
    
    
    dataset_training = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % len(dataset_training))

    model = create_model(opt)      # create a model given opt.model and other options

    # create a validation dataset given opt.dataset_mode and other options
    opt.name = opt.name + "_" + opt.model+  "_" + opt.netG + "_" + opt.netD + "_" + str(opt.n_layers_D) + "_" + str(opt.load_size) +"_" + str(opt.crop_size)+ "_" + opt.preprocess

    opt.phase = 'val'
    dataset_eval = create_dataset(opt)      # create a dataset for evaluations
    opt.num_test = len(dataset_eval)
    opt.phase = 'train'
    print('The number of eval images = %d' % len(dataset_eval))
    # initiate  inception for calculating FID score
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model=inception_model.cuda()



    logger = None
    if log:
        logger = Logger_wb(opt) # create the wandb logger
    else:
        warnings.warn("Logger is of")
        print("------------------- -------------- --------------------")
        print("------------------- LOGGING IS OFF --------------------")
        print("------------------- -------------- --------------------")
    
    train(model, dataset_training, dataset_eval, inception_model, opt, logger)

    if log:
        logger.finish()
    