import sys
sys.path.append(".")
from main import *
from config_act import Config

cfg=Config('volleyball')
cfg.input_size = 224, 224
cfg.image_size = 256, 320
cfg.use_gpu=True
cfg.device_list="0"
cfg.training_stage=1
cfg.stage1_model_path=None
cfg.train_backbone=True
cfg.data_path='/media/nas-570-002-nfssamba/rsanford/activity_recognition/volleyball/'

cfg.batch_size=8
cfg.test_batch_size=4
cfg.num_frames=6
cfg.num_before=3
cfg.num_after=2
cfg.train_learning_rate=0.01
cfg.lr_plan={41:0.001, 81:0.001, 121:0.0001}
cfg.max_epoch = 200
cfg.test_interval_epoch = 5
cfg.test_before_train = False
cfg.out_size = 64, 64
cfg.crop_size = 3, 3
cfg.exp_name = 'resnet_gwnet'
cfg.train_dropout_prob = 0.2
cfg.weight_decay = 0.0001
cfg.dropout = True
cfg.non_local = False
cfg.stgcn = False
cfg.gwnet = True

cfg.exp_note='Volleyball_stage1'
main(cfg, need_new_folder=True)