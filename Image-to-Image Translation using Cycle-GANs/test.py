import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html


def transfer(dataroot, G_path, direction, suffix, result_dir):
    
    opt = TestOptions()  # get test options
    # hard-code some parameters for test
    opt.dataroot = dataroot
    opt.name = G_path
    opt.model = 'test'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    opt.dataset_mode = 'unaligned'
    opt.phase = 'test'
    opt.direction = direction
    opt.input_nc = 3
    opt.output_nc = 3
    opt.preprocess = 'resize_and_crop'
    opt.load_size = 286
    opt.crop_size = 256
    opt.isTrain = False
    opt.gpu_ids = '0'
    opt.max_dataset_size = float("inf")
    opt.checkpoints_dir = 'model_checkpoints'
    opt.model_suffix = suffix
    opt.ngf = 64
    opt.ndf = 64
    opt.netG = 'resnet_9blocks'
    opt.norm = 'instance'
    opt.no_dropout = True
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.load_iter = 0
    opt.epoch = 'latest'
    opt.verbose = False
    opt.results_dir = result_dir
    opt.eval = False
    opt.serial_batches = False
    opt.num_test = 3
    opt.aspect_ratio = 1.0
    opt.display_winsize = 256
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
