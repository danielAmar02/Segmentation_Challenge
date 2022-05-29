# Challenge on Land Cover Predictive Model


Land cover mapping is a fundamental problem in Earth observation. Land cover is the observed physical material covering the Earth's surgace. Accurately predicting the physical materials in an environment has many applications, such as analysing the development of crops in agriculture and modeling the distribution of species in ecology.

![alt text](https://github.com/danielAmar02/challenge-ens/blob/main/Media/pres.jpg)

Training/Validation dataset: 18,491 (256x256 pixel) images of 4 bands (RGB-NIR) extracted from the Sentinel-2 sattelite. For each image, we have an associated Mask for the training dataset

Test Dataset: 5000 images

Evaluation metric : Kullback-Leibler divergence 


The goal of the challenge is to do segmentation on satellite images and to tell given an image the proportion of pixels for 8 classes (artificial, cultivated, broadleaf, coniferous, herbaceous, natural, snow, water). As expected, the dataset is higly imbalanced.



# Results 

We used a Unet++ in order to reconstruct masks from the images and then calculated from those masks the proportion of pixels per class. Unet++ keeps the general architecture of the Unet but the encodeur and the decodeur are now linked by convolutional blocks. 


![alt text](https://github.com/danielAmar02/challenge-ens/blob/main/Media/Unet%2B%2B.jpg)



In order to deal with the unbalanced dataset, we used the Dice Loss and the Weighted Cross Entropy loss wich allows us to have the following results



| Model | Kullback-Leibler divergence |
|:-----:|:-----:|
|Random|0.63|
|Soft classification CNN| 0.23|
|Unet++ WCE loss| 0.1 |
|Unet++ Dice loss | 0.085 |



![alt text](https://github.com/danielAmar02/challenge-ens/blob/main/Media/results.jpg)
