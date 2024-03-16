# PAM-FDnet
Paper: Noise-insensitive defocused signal and resolution enhancement for optical-resolution photoacoustic microscopy via deep learning
Authors: Rui Wang, Zhipeng Zhang, Ruiyi Chen, Xiaohai Yu, Hongyu Zhang, Gang Hu, Qiegen Liu, Xianlin Song
J Biophotonics. 2023 Oct;16(10):e202300149.
https://doi.org/10.1002/ibio.202300149

Date : mar-16-2024
Version : 1.0
The code and the algorithm are for non-comercial use only.
Copyright 2024, Department of Electronic Information Engineering, Nanchang University.

<div align="justify">
Optical-resolution photoacoustic microscopy suffers from narrow depth of field and a significant deterioration in defocused signal intensity and spatial resolution.
Here, a method based on deep learning was proposed to enhance the defocused resolution and signal-to-noise ratio. A virtual optical-resolution photoacoustic microscopy based on k-Wave was used to obtain the datasets of deep learning with different noise levels.
A fully dense U-Net was trained with randomly distributed sources of different shapes to improve the quality of photoacoustic images. 
The results show that the PSNR of defocused signal was enhanced by more than 1.2 times.An over 2.6-fold enhancement in lateral resolution and an over 3.4-fold enhancement in axial resolution of defocused regions were achieved. The large volumetric and
high-resolution imaging of blood vessels further verified that the proposed method can effectively overcome the deterioration of the signal and the spatial resolution due to the narrow depth of field of optical-resolution photoacoustic microscopy. 
</div>

# Virtual OR-PAM for dataset acquisition.
![image](https://github.com/yqx7150/FD-Unet/assets/26964726/693a20ab-dc6d-40f7-af10-76c2e20139b3)

# Diagram of the FD-U-Net architecture.
![image](https://github.com/yqx7150/FD-Unet/assets/26964726/2f721244-e893-4ca3-a0bc-af297e6cdf1d)
