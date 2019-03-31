# Our 2D-Assisted Self-supervised Learning (2DASL) is available at: https://arxiv.org/abs/1903.09359

To facility the research in the community of 3D face reconstruction and 3D face alignment, we will release our source code, including the pytorch code for testing (training code will be available upon the acceptance of our paper), the matlab code for 3D plot, 3D face rendering and evaluation. The models and demo will be released soon.

# We add the matlab evaluation code with the metric Normalized Mean Error (NME), specifically,
  1. We compare the results of our 2DASL with PRNet on the sparse 68 key points on both 2D&3D coordinates.  
  2. We compare the results of our 2DASL with 3DDFA on all points for the task of dense alignment (both 2D and 3D coordinates) and
     3D face recnostruction.
  3. The Iterative Closest Point (ICP) algorithm, which is used to find the nearest points between two 3D models is included in 
     our codes.
     
  Usage: Download the "visualize.rar" package at： https://pan.baidu.com/s/1M0sg3eqrtzxxMJqmAzczRw using the password: "h26t".
         Putting the package under the folder "evaluation", extracting it and then run the "nme_for_*" scripts. 

