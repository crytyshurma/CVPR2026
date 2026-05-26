This software is a part of supplementary material of a CVPR2026 paper: 

    Tomoya Tsuchida, Keita Takahashi, Chihiro Tsutake, Toshiaki Fujii, Hajime Nagahara: 
    "Coded-E2LF: Coded Aperture Light Field Imaging from Events", 
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026.

When you use the codes for research purpose, please include citation to the above paper in your written documents and presentations.


Overview
-----------------------------------------------------

This software can reconstruct a light field from the event data captured with a coded aperture camera. The pre-trained network weights (Baseline+BF+RA model) are also included. Both a light field (Lego Knights, provided in BasicLFSR test dataset [1]) and real-world event data (Animals, Fish) can be used as test data.

    [1] Zhengyu Liang. BasicLFSR (open source light field toolbox for super-resolution). https://github.com/ZhengyuLiang24/BasicLFSR, 2021.

Requirement
-----------------------------------------------------
This software requires Python (version 3.13.5), PyTorch (version 2.7.1), and CUDA (version 12.8). Our python environment is shown in the requirements.txt. We tested the software on a PC equipped with an NVIDIA RTX 5090, running Ubuntu 22.04.5 LTS. 


Usage
-----------------------------------------------------
** 1. Installation **

To install the dependencies, run:
    > pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    > pip3 install -r requirements.txt


** 2. Run a test from a light field **

We include a source light field (8x8 views, Lego Knights), which are located at:
    - LF_data/8x8/BasicLFSR/Lego_Knights/

Execute "test_from_LF.py" as follows.
    > python3 test_from_LF.py
    
The software first generates event images from the source light field, simulating the behavior of an event camera.
The software then reconstructs a light field from the event images.

The outputs are stored at:
    - Dst/test/from_LF_data/BasicLFSR/Lego_Knights/acquired_data: acquired (simulated) event images
    - Dst/test/from_LF_data/BasicLFSR/Lego_Knights/re_views: reconstructed light field (8x8 views)


** 3. Run a test from event data **

We also include the event data captured from two real scenes (Animals, Fish) by using our imaging system, located at:
    - Acquired_data/Animals
    - Acquired_data/Fish 

Each folder contains:
    - Event images used as input to the software (event_image_00000.npy, event_image_00001.npy, event_image_00002.npy)
    - Event images for visualization (event_image_vis_00000.png, event_image_vis_00001.png, event_image_vis_00002.png)
    
Note that *.npy files and *.png files are equivalent in terms of the information contained.

To reconstruct light fields (8x8 views) from the event data, execute "test_from_acquired_data.py" as follows.
    > python3 test_from_acquired_data.py

The outputs (light fields with 8x8 views) are stored at:
    - Dst/test/from_acquired_data/Animals
    - Dst/test/from_acquired_data/Fish
    




-----------------------------------------
## License
-----------------------------------------
The MIT License (MIT)

Copyright (c) 2026 Tomoya Tsuchida at Fujii Lab. of Nagoya University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
