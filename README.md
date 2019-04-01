# Our 2D-Assisted Self-supervised Learning (2DASL) is available at: https://arxiv.org/abs/1903.09359 
This project is created by Tu Xiaoguang (xguangtu@outlook.com) and Luo Yao (luoyao_alpha@outlook.com)

To facility the research in the community of 3D face reconstruction and 3D face alignment, we will release our source code, including the pytorch code for testing (training code will be available upon the acceptance of our paper), the matlab code for 3D plot, 3D face rendering and evaluation. The models and demo will be released soon.

# We add the matlab evaluation code with the metric Normalized Mean Error (NME) in the folder "evaluation", including:
  1. We compare the results of our 2DASL with PRNet on the sparse 68 key points on both 2D&3D coordinates.  
  2. We compare the results of our 2DASL with 3DDFA on all points for the task of dense alignment (both 2D and 3D coordinates) and
     3D face recnostruction.
  3. The Iterative Closest Point (ICP) algorithm, which is used to find the nearest points between two 3D models is included in 
     our codes.
     
  Usage: Download the "visualize.rar" package at： https://pan.baidu.com/s/1M0sg3eqrtzxxMJqmAzczRw using the password: "h26t".
         Putting the package under the folder "evaluation", extracting it and then run the "nme_for_*" scripts. 
         
# We add our test codes in the folders "models" and "test_codes", including:
  1. The 2DASL modles for stage1 and stage2, in the folder "models".
  2. The test codes to obtain 3D vertex for ALFW2000-3D images.
  3. The codes to plot 2D facial landmarks.
  
  Usage: python-3.6.6, pytorch-0.4.1. Just run the script "main_visualize_plot_4chls.py" in the folder "test_codes".
  
# We add our 3D face rendering codes in the folder "3D_results_plot", including:
  1. The codes for 3D face rendering, i.e., drawing 3d reconstruction mask.
  2. The codes for plotting dense face alignment.
  3. The codes for writing .obj file, the .obj files could be opend by the software meshlab. 
  
  Usage: Download the "visualize.rar" package at： https://pan.baidu.com/s/1M0sg3eqrtzxxMJqmAzczRw using the password: "h26t".
         Putting the package under the folder "3D_results_plot", extracting it and then run the "test_good_byXgtu.m*" scripts. 
         
 # Citation
  If you find our code is useful for your research, pls cite our work:
  
@article{tu2019joint,
  title={Joint 3D Face Reconstruction and Dense Face Alignment from A Single Image with 2D-Assisted Self-Supervised Learning},
  author={Tu, Xiaoguang and Zhao, Jian and Jiang, Zihang and Luo, Yao and Xie, Mei and Zhao, Yang and He, Linxiao and Ma, Zheng and Feng, Jiashi},
  journal={arXiv preprint arXiv:1903.09359},
  year={2019}
}

