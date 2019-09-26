import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision

from config_act import *

# from model import GGCN
from model_2d import GGCN
from metric import accuracy

from dataset import return_dataset

from tensorboardX import SummaryWriter

from st_gcn import Model

from utils import *
from collections import OrderedDict

import torch.nn as F

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def to_cpu(model_state):
    new_state = OrderedDict()
    for k, v in model_state.items():
        new_state[k] = v.cpu()
    return new_state

def save_checkpoint(epoch, model, optimizer, save_path, cpu):
    if cpu:
        model_state = to_cpu(model.state_dict())
    else:
        model_state = model.state_dict()

    checkpoint = {
        'next_epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict(),
        # 'scheduler_state': scheduler.state_dict(),
    }

    torch.save(checkpoint, save_path)

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['next_epoch']
    model_state = checkpoint['model_state']
    optimizer_state = checkpoint['optimizer_state']
    # Setup model
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return model, optimizer, start_epoch

def main(cfg, need_new_folder=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config(need_new_folder=need_new_folder)
    show_config(cfg)
    tf_writer = SummaryWriter(log_dir=cfg.result_path)

	# Get training and validation set
    training_set, validation_set = return_dataset(cfg)

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 4
    }
    training_loader = data.DataLoader(training_set, **params)
    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)

    # Initialize model
    model = GGCN(cfg)
	
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device=device)
    else:
        device = None

    optimizer = optim.SGD(model.parameters(), lr=cfg.train_learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
    start_epoch = 1
    # Load model
    if cfg.stage1_model_path is not None:
        print('Loading previous model ' + cfg.stage1_model_path)
        model, optimizer, start_epoch = load_checkpoint(cfg.stage1_model_path, model, optimizer)

    best_result = {'epoch': 0, 'activities_acc': 0}
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        # One epoch of forward and backward
        train_info = train(training_loader, model,
                           device, optimizer, epoch, cfg, tf_writer)
        show_epoch_info('Train', cfg.log_path, train_info)

        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model, device, epoch, cfg,  tf_writer)
            show_epoch_info('Test', cfg.log_path, test_info)

            if test_info['activities_acc'] > best_result['activities_acc']:
                best_result = test_info
            print_log(cfg.log_path,
                      'Best group activity accuracy: %.2f%% at epoch #%d.' % (best_result['activities_acc'], best_result['epoch']))

            # Save model
            if cfg.training_stage == 2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + \
                    '/stage%d_epoch%d_%.2f%%.pth' % (
                        cfg.training_stage, epoch, test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:', filepath)
            elif cfg.training_stage == 1:
                filepath = cfg.result_path + \
                    '/stage%d_epoch%d_%.2f%%.pth' % (
                        cfg.training_stage, epoch, test_info['activities_acc'])
                save_checkpoint(epoch, model, optimizer, filepath, True)
            else:
                assert False

def qc_image(input_image):
    output_image = torchvision.utils.make_grid(input_image, normalize=True, scale_each=True)
    return output_image

def train(data_loader, model, device, optimizer, epoch, cfg, tf_writer):
    ce_loss = F.CrossEntropyLoss()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    for i, batch_data in enumerate(data_loader):
        model.train()
        # prepare batch data
        if device is not None:
            batch_data = [b.to(device=device) for b in batch_data]

        input_data = batch_data[0]
        bbox_data = batch_data[1]
        activities_in = batch_data[2]

        batch_size = input_data.shape[0]
        num_frames = input_data.shape[1]

        # Write one batch to tensorboard.
        tf_writer.add_image('bbox', qc_image(input_data[0]), epoch*i)

        input_data = torch.reshape(input_data, (-1, 3, cfg.out_size[0], cfg.out_size[1]))

        # activities_in = activities_in.reshape((batch_size, num_frames))

        activities_in = activities_in[:, 0].reshape((batch_size,))

        # forward
        # input_data = input_data.permute((0, 2, 1, 3, 4))

        if cfg.stgcn:
            activities_scores = model((input_data, bbox_data))
        else:
            activities_scores = model(input_data)

        # Predict activities
        activities_loss = ce_loss(activities_scores, activities_in)
        activities_labels = torch.argmax(activities_scores, dim=1)
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

        # Get accuracy
        activities_accuracy = activities_correct.item() / \
            activities_scores.shape[0]

        activities_meter.update(activities_accuracy,
                                activities_scores.shape[0])

        # Total loss
        loss_meter.update(activities_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        activities_loss.backward()
        optimizer.step()

    tf_writer.add_scalar('loss/train', loss_meter.avg, epoch)
    tf_writer.add_scalar('acc/activities_acc', activities_meter.avg*100, epoch)
    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg*100,
    }

    return train_info


def test(data_loader, model, device, epoch, cfg, tf_writer):
    model.eval()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    ce_loss = F.CrossEntropyLoss() 
    epoch_timer = Timer()
    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            if device is not None:
                batch_data_test = [b.to(device=device) for b in batch_data_test]
            input_data = batch_data_test[0]
            bbox_data = batch_data_test[1]
            activities_in = batch_data_test[2]

            batch_size = input_data.shape[0]
            num_frames = input_data.shape[1]

            input_data = torch.reshape(input_data, (-1, 3, cfg.out_size[0], cfg.out_size[1]))

            if cfg.stgcn:
                activities_scores = model((input_data, bbox_data))
            else:
                activities_scores = model(input_data)
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # Predict activities
            activities_loss = ce_loss(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)

            # actions_correct = torch.sum(
                # torch.eq(actions_labels.int(), actions_in.int()).float())
            activities_correct = torch.sum(
                torch.eq(activities_labels.int(), activities_in.int()).float())

            # Get accuracy
            # actions_accuracy = actions_correct.item()/actions_scores.shape[0]
            activities_accuracy = activities_correct.item() / \
                activities_scores.shape[0]

            # actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(
                activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss = activities_loss
            loss_meter.update(total_loss.item(), batch_size)

    tf_writer.add_scalar('loss/test', loss_meter.avg, epoch)
    tf_writer.add_scalar('acc/test_activities_acc', activities_meter.avg*100, epoch)
    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg*100,
    }

    return test_info