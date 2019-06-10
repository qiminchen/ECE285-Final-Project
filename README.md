# ECE285-Final-Project
ECE 285 – MLIP – Project B Style Transfer

# Description
This is project B Style Transfer developed by team 996ICU composed of Qimin Chen, Yunfan Chen and Ke Xiao. Our group aim at implementing the style transfer by replicating the experiment proposed by Gatys et al. in 2015. We also implement the style transfer by using Cycle-GAN introduced by Jun Yan et al. Then we slightly modify the cycle consistency loss to force generator to focus more on the content structure in order to produce more realistic images.

# Requirement
Install package 'dominate' as: ```pip install dominate --user```     
Install package 'visdom' as: ```pip install visdom --user```

If ```cuda runtime error (11) : invalid argument at /opt/conda/conda-bld/pytorch_1544174967633/work/aten/src/THC/THCGeneral.cpp:405``` happened, please restart the GPU cluster.

# Code organization
<pre>
Neural Style Transfer     -- Folder contains code for neural style transfer proposed by Gatys et al.
    
    demo.ipynb            -- Run a demo of our code
    train.ipynb           -- Run an example of our code
    utilis.py             -- Module of model and dataset
    visu.py               -- Module of visualize images and results

Image-to-Image Translation using Cycle-GANs  -- contains code for Cycle-GANs introduced by Jun Yan et al.

    data                  -- Folder contains dataset.py etc.
    model_checkpoints     -- Folder contains trained cycle-gan model and trained improved cycle-gan model
    model                 -- Folder contains model.py and network.py etc.
    options               -- Folder contains train and test configuration
    transferred_data_for_recover  -- Folder contains transfered images for recovering
    util                  -- Folder contains utilities
    demo.ipynb            -- Run a demo of our code
    test.py               -- Test module
    train.py              -- Train module
    utilis.py             -- Utilities for visualization
    
<pre>
