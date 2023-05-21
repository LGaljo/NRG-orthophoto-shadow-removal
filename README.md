## 2.4 Removing shadows from ortho-photo images
The goal of this seminar is to implement an algorithm that will take an aerial ortho-photo as an input and output an
image with removed shading/shadows. To achieve this, you may apply one of the algorithms presented in [1, 2] or
some other state-of-the-art shadow removal algorithm. The developed algorithm should be able to read images in
different formats (at least JPG and PNG) and output the image in the same format. The application should support
piping, so it can be used in an image processing pipeline.

### Implementation
This implementation is based on ST-CGAN and paper Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal [3],
Model has been trained on ISTD dataset which provides images with/without shadows and shadow masks for around 5000 epochs.
The ortho-photo images are fetched from ARSO GIS viewer using custom download script.
They are then merged into one and shown in viewer.

The shadow removal process image slices into multiple tiles of 256x256 pixel.
Model processes each tile and then restitches them into single image.

The GUI allows comparison of original and transformed image by switching between them.

We expanded the ISTD dataset and additional images are available on [Google Drive](https://drive.google.com/drive/folders/1Z17594slIvxKwVPuH3KccTMnIHDnK3gB?usp=share\_link).
This folder also includes learned model.

### How to use?
* Download the model from the Google Drive and place it into the folder at ./st_cgan/model.
* Run python script ./main.py to open the GUI

If you want to retrain the model use the training scripts provided in the [repo of the original authors implementation of research paper](https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks
).


### Thanks to
This StackOverflow answer for advanced image viewer implementation Tkinter:
https://stackoverflow.com/a/48137257/16927038
GitHub repository of research paper [3], which hosts the model and training scripts to create the model:
https://github.com/IsHYuhi/ST-CGAN_Stacked_Conditional_Generative_Adversarial_Networks

### References
[1] S. H. Khan, M. Bennamoun, F. Sohel and R. Togneri, "Automatic Shadow Detection and Removal from a Single
Image", IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016, doi:
10.1109/TPAMI.2015.2462355.  
[2] G. D. Finlayson, S. D. Hordley, M. S. Drew, "Removing Shadows from Images", Proceedings of the 7th
European Conference on Computer Vision, 2002, url: https://dl.acm.org/doi/10.5555/645318.649239 
[3] Jifeng Wang∗, Xiang Li∗, Le Hui, Jian Yang, Nanjing University of Science and Technology
Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal, url: https://arxiv.org/abs/1712.02478
