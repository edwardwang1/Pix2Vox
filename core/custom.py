# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

def custom_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None, \
        test_writer=None, encoder=None, decoder=None, refiner=None, merger=None):
    #Setting outputDir
    output_dir = cfg.DIR.OUT_PATH
            
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True


    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    rendering_image_paths = []

    for folder, subs, files in os.walk('/home/eprisman/projects/def-eprisman/eprisman/CustomRenderingImages/mand1/'):
        for filename in files:
            rendering_image_paths.append(os.path.abspath(os.path.join(folder, filename)))

    print("number of files", len(rendering_image_paths))

    selected_rendering_image_paths = [rendering_image_paths[i] for i in range(len(rendering_image_paths))]
    #selected_rendering_image_paths = [rendering_image_paths[i] for i in range(cfg.CONST.N_VIEWS_RENDERING)]

    rendering_images = []
    for image_path in selected_rendering_image_paths:
        rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if len(rendering_image.shape) < 3:
            print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                  (dt.now(), image_path))
            sys.exit(2)

        rendering_images.append(rendering_image)

    rendering_images = np.asarray(rendering_images)

    with torch.no_grad():
        # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)

        # Test the encoder, decoder, refiner and merger
        image_features = encoder(rendering_images)
        raw_features, generated_volume = decoder(image_features)

        if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)

        if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
            generated_volume = refiner(generated_volume)

    print(type(generated_volume))
