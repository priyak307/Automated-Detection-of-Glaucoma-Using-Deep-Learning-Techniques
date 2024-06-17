# Automated-Detection-of-Glaucoma-Using-Deep-Learning-Techniques
Data Collection: Gathering the dataset is the initial step in the approach. The G1020 dataset, which is open to the public, is used. The dataset consists of 1020 fundus images, including 600 normal and 420 glaucomatous images.

Prep-processing: To improve the contrast of the optic disc and cup, Contrast Limited Adaptive Histogram Equalizer (CLAHE) is used in which the L-channel of the Lab colour space is extracted and combined with the A and B channels. This pre-processing phase is crucial to increasing the precision of segmentation and subsequent glaucoma identification.

Segmentation: A well-liked deep learning architecture called U-Net is employed for medical image segmentation. Bottleneck double convolution is used into the modified U-Net architecture in this study to increase the segmentation accuracy for glaucoma detection. Transposed convolution is used to boost the resolution of feature maps, and the model is capable of learning intricate representations of input images. The segmentation accuracy for the modified U-Net model employed in this investigation is 97%.

Feature Extraction: The cup area, disc area, CDR, and rim width are extracted from the images. These features are extracted using image processing techniques and are frequently employed in the automated detection of glaucoma.

Model: The VGG-19 architecture, a well-liked architecture for image recognition tasks, served as the foundation for the deep learning model employed in this work. The VGG-19 architecture's fully connected layer is replaced with long short-term memory (LSTM) layers to increase computational efficiency. The model is developed using a processed image, and after that, statistical features are extracted. A machine learning model is created in addition to the deep learning model. It employs support vector machines (SVM) and logistic regression. The machine learning model is trained using the four extracted characteristics, which are the cup area, disc area, CDR, and rim width.

Pipelining: The deep learning and machine learning models are combined using pipelining to increase the precision of the glaucoma detection model. The best features and methods from the deep learning and machine learning models are combined to create the final model.

Evaluation: Accuracy curves like area under the curve (AUC) and receiver operating characteristic (ROC) are used to assess how well the glaucoma detection model performs. To evaluate the effectiveness of the model and choose the best threshold for classification, the AUC and ROC curves are utilised.
![image](https://github.com/priyak307/Automated-Detection-of-Glaucoma-Using-Deep-Learning-Techniques/assets/104674161/41c9a279-171e-4e3c-85dc-4a0e3c5800db)
