# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Image Caption



## Problem Statement

"A picture is worth a thousand words". This adage was first coined in the 1900s. The idea behind it was that complex and sometimes multiple ideas can be conveyed by a single still image.

However, for the visually impaired, is there a way to interpret these images so that they are able capture the essence of what the image is portraying?

In this capstone, I will be exploring the use of Neural Networks to generate captions to describe an image. Thereafter, with the help of an API, provide an audio output of the generated captions.


## Datasets

Data used in this modelling are:

- `Images`: Folder that contains 8,091 photographs from Flickr
- `Random_Images`: Folder that contains 8 unseen images
- `caption.txt`: contains 40,455 captions that correspond to the photographs in the Images Folder. There are 5 captions to describe each photograph. Each caption is idenified by the image path

## Contents
This project is split into several notebooks.

- Part 1: Data Exploration
- Part 2a: Feature Extraction (VGG16)
- Part 2b: Feature Extraction (Inception v3)
- Part 3a: Preprocessing and Modelling (VGG16 Base Model)
- Part 3b: Preprocessing and Modelling (VGG16 Base Model + Dropout)
- Part 3c: Preprocessing and Modelling (VGG16 Base Model + Dropout + Kernel Regularizer)
- Part 4a: Preprocessing and Modelling (Inception v3 Base Model)
- Part 4b: Preprocessing and Modelling (Inception v3 Base Model + Dropout)
- Part 4c: Preprocessing and Modelling (Inception v3 Base Model + Dropout + Kernel Regularizer)
- Part 5: Image Caption for Random Images Plus Audio Output

    
## Modelling Process
The follow steps were taken:

1. Exploratory Data Analysis
2. Feature Extraction
3. Preprocessing / Feature Engineering
4. Hyperparameter Tuning / Modelling
5. Results
6. Summary and Improvements

## Exploratory Data Analysis

Exploratory Data Analysis was performed on the captions provided as part of the dataset. There are a total of 40,455 captions for 8,091 images, which translates to 5 captions per image. 

#### Overview of the images that correspond to the captions 
<img src = "assets/overview.png" width = "800">

#### Distribution of Word Counts
After cleaning, word count per caption decreased but the frequency of the median word count has increased

<img src = "assets/word count dist before.png" width = "500">

<img src = "assets/word count dist after.png" width = "500">

#### 50 Most Frequent Occurring N-grams

Based on the charts below, you can tell that the images consists of a lot of dogs and humans (people, man, boy, woman, girl). In terms of phyiscal objects, they do not appear anywhere in the top 50 unigram or bigram.

This probably gives some indication that the model will not be able to predict objects very well.

<img src = "assets/50 unigram.png" width = "800">

<img src = "assets/50 bigram.png" width = "800">

## Feature Extraction

Two pre-trained CNN models were used to perform feature extraction in this project, and they are: VGG16 and Inception v3. Both models were pre-trained using the ImageNet dataset.

VGG16 is 16-layers deep while Inception v3 is 48 layers deep. In both cases, the prediction layers were removed.

The extracted features are saved in pickle files to be used in the modelling.

- [VGG16 Features](code/features_vgg16.pkl)
- [Inception v3 Features](code/features_inceptionv3.pkl)

## Preprocessing / Feature Engineering

Cleaning was performed on the captions to convert the text to lowercase, and to remove digits and special characters. Start "startseq" and End "endseq" tags were added to each caption to signify the start and end of each caption

|X1 (feature vector)   |   X2 (text sequence)                |  y (word to predict) |
|:--------------------:|-------------------------------------|:--------------------:|
|Feature               |  startseq,                          |  two                 |
|Feature               |  startseq, two                      |  dogs                |
|Feature               |  startseq, two, dogs                |  drink               |
|Feature               |  startseq, two, dogs, drink         |  water               |
|Feature               |  startsrq, two, dogs, drink, water  |  endseq              |

The captions were then mapped to their corresponding images.

## Hyperparameter Tuning / Modelling

Captioning Models were created using previously extracted features. The hyperparameter tuning are as follows:

- VGG16 Base Model with a standard Dropout of 0.4
- VGG16 Base Model with increased Dropout of 0.6
- VGG16 Base Model with increased Dropout of 0.6 and Kernel Regularizer using Ridge Regression, lambda = 0.01
- Inception v3 Base Model with a standard Dropout of 0.4
- PInception v3 Base Model with increased Dropout of 0.6
- Inception v3 Base Model with increased Dropout of 0.6 and Kernel Regularizer using Ridge Regression, lambda = 0.01


#### VGG16

<img src = "assets/model vgg16.png" width = "500">

#### Inception v3 
<img src = "assets/model inception v3.png" width = "500">

The parameters of the Captioning Models training are as follows:

|Parameter                   |   Value                           |
|----------------------------|-----------------------------------|
|Epochs                      |   300                             |
|Batch Size                  |   32                              |
|Early stopping (patience)   |   3 epochs (drop in val accuracy) |
|Optimizer                   |   Adam                            |
|Learning Rate               |   0.001                           | 
|Loss                        |   Categorical Crossentropy        |
|Metrics                     |   Accuracy                        |

## Results

### Performance Evaluation

After including higher Dropout and introducing Kernel Regularizer, both Captioning models generalised better. However, the accuracy drops by about 0.1 with each modification.

#### Results for VGG16 Base Model (Dropout = 0.4)

<img src = "assets/acc chart base vgg model.png" width = "500">

<img src = "assets/loss chart base vgg model.png" width = "500">

#### Results for VGG16 Base Model + Dropout = 0.6

<img src = "assets/acc chart base vgg plus dropout model.png" width = "550">

<img src = "assets/loss chart base vgg plus dropout model.png" width = "550">

#### Results for VGG16 Base Model + Dropout = 0.6 + Kernel Regularizer (L2 = 0.01)

<img src = "assets/acc chart base vgg plus dropout kernel regularizer model.png" width = "650">

<img src = "assets/loss chart base vgg plus dropout kernel regularizer model.png" width = "650">

#### Results for Inception v3 Base Model (Dropout = 0.4)

<img src = "assets/acc chart base inception model.png" width = "500">

<img src = "assets/loss chart base inception model.png" width = "500">

#### Results for Inception v3 Base Model + Dropout = 0.6

<img src = "assets/acc chart base inception plus dropout model.png" width = "550">

<img src = "assets/loss chart base inception plus dropout model.png" width = "550">

#### Results for Inception v3 Base Model + Dropout = 0.6 + Kernel Regularizer (L2 = 0.01)

<img src = "assets/acc chart base inception plus dropout kernel regularizer model.png" width = "650">

<img src = "assets/loss chart base inception plus dropout kernel regularizer model.png" width = "650">

#### (Bilingual Evaluation Understudy) BLEU Score

BLEU assigns a single numerical score to a machine generated translation that tells you how good it is compared to one or more reference translation.

The scores are calculated using modified n-gram precision score that clips the number of times to count a word, based on the maximum number of time it appears in the reference translation.

BLEU-1 corresponds to Unigram, BLEU-2 corresponds to Bigram and so on. The individual scores are calculated also by weighting them by calculating the Weighted Geometric Mean.

There is also a brevity penalty imposed on sentences that are too short and punishment is not imposed on long sentences.

One thing to note, BLEU does not take into account the order in which the words appear in the translation.

  
|       | VGG16 Base    |   VGG16 Base + Dropout  | VGG16 Base + Dropout + Kernel Regularizer     |     Inception v3 Base  |   Inception v3 Base + Dropout  |   Inception v3 Base + Dropout + Kernel Regularizer   |
|:-------:|:------:|:------:|:-----:|:-----:|:-----:|:-------:|
|   BLEU-1    |    0.65|0.62|0.57|0.59|0.58|0.52    |
|  BLEU-2     |     0.47|0.43|0.33|0.39|0.38|0.28|
|  BLEU-3     |     0.36|0.32|0.21|0.27|0.26|0.17|
|  BLEU-4     |     0.28|0.24|0.12|0.19|0.18|0.1|    

------------
### Visualisation Evaluation

#### VGG16

<img src = "assets/results1.png" width = "800">

<img src = "assets/results2.png" width = "800">

#### Inception v3

<img src = "assets/results3.png" width = "800">

<img src = "assets/results4.png" width = "800">

------------

### Caption Unseen Images

#### Snow

<img src = "assets/unseen1.png" width = "800">

#### Solider

<img src = "assets/unseen2.png" width = "800">

#### Dog

<img src = "assets/unseen3.png" width = "800">

------------

### Audio Output

Google Text-to-Speech (gTTS) API was used to provide audio output to the what I would consider to be best predicted captions. The outputs for the unseen images are in the following files:

- [Snow](code/snow.wav)
- [Soldiers](code/soldiers.wav)
- [Dog](code/dog.wav)

 
## Summary and Improvements

### Summary

- EDA of captions gave a good understanding of the images that are in the dataset and what to expect in the results
- Some of the captions are not useful in training the model
- Overfitting is greatly reduced after introducing Dropout and Kernel Regularizer
- Improving the generalisation of the models resulted in a decrease in overall accuracy
- Training of models took a long time and depends a lot on luck


### Improvements

- Available applications (for Transfer Learning) on TF Keras are trained on ImageNet dataset. To explore using Yolo pre-trained model and see if the overall results perform better than the current one (as Yolo is trained using MS Coco dataset) 
- Extract features using a larger dataset
- Test out more values in terms of the Dropout Rate and Kernel Regularizer and its Lambda value and manipulate different batch sizes, learning rate, train/test split values in the training process
- Try out different activation functions e.g. ReLU vs GeLU/Swish Activation Functions
- Apply other Evaluation Metrics like METEOR/ROUGE

