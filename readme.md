#### This document is intended to be a complement to the video.

## Who we are
Hi! I'm Stanley, SDE Intern at Kings Distributed Systems/Distributed Computer.
I mainly work on ML tasks, and spend my free time on Kaggle - Google's data science competitions platform, where I am ranked 1100th globally - the majority of my experience is in medical imaging.

## Inspiration
We were inspired by an app: [skinvision.com](https://skinvision.com). 
They charge $10 per scan of lesions. 
While we don't know what methods they are using, we can read about the academic State of the Art through papers.

Melanoma detection is a relatively "done to death" topic, and one that is rarely ever done properly, especially in hackathons. 
This document aims to explain our reasoning behind design decisions and their quantitate impact.
Our application is based on a paper [1] detailing the approach to the State of the Art in skin cancer detection by three Nvidia data scientists I stumbled upon while on ArXiV. 
Further details in "How we built it" below.

## What it does
It diagnoses skin cancer, among other conditions. 
Our application works in three stages. 
First, the user takes a portrait for skin tone detection - the American Cancer Society notes melanoma is 20 times more common in Caucasian individuals than African American.

Next, the user inputs simple patient data - their age, sex, and the lesion location.
Finally, the user uploads an image of the lesion itself.

All three of these inputs are taken into account for the final melanoma risk factor. 
Our models also output the most likely diagnosis of a few common skin conditions that are commonly confused for melanoma. 
These are Actinic Keratosis, Basal Cell Carcinomaa, Benign Keratosis, Dermatofibroma, Squamous Cell Carcinoma, Vascular Lesions, and Nevus.


## How we built it (machine learning)
### Modelling approach
This was an elaborate one! I'm very proud of the modelling approach.
Our model is an ensemble of various efficientnet models at varying image sizes and model sizes.
This is a fairly standard approach for ensembling, but what sets our models appart is our two head system.

Instead of using metadata with a standard tabular model, I read an approacah of using a dual head model from Ha, Q., et al.
One head was a standard CNN head - with an image as input, and the second head was a tabular model with metadata as input.
Then, they were concatenated before the classification layers.
I didn't have time to run a before and after, but a rough estimate would be about a 5-15% performance uplift compared to a CNN on its own.
A tabular model ensembled with a CNN on its own also doesn't perform as well.

On a more boring aspect of hyperparameter tuning, we were very lucky to have significant compute from both local GPU's as well as AWS.
Our models trained on 2x Nvidia RTX3090 as well as 8x Nvidia Tesla V100 on AWS. Hyperparameter tuning was done manually.
Augmentations were also taken from Ha, Q., et al.
We debated whether to use test time augmentation (TTA) but finally decided not to, as an ensemble was more meaningful than TTA.
### Metric design
In order to focus on reliable detection with an emphasis on low false negative rate, we evaluated our models on weighted area under curve (wAUC).
The definition of a false negative is when the output of the model is negative, but the actual outcome is positive.
This is detrimental in melanoma diagnosis - so we created a metric to minimize it.
We weight the false negative weight as double that of the false positive weight, as follows

```
fpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```
In other words, the area between the false negative rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). 
The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

This does, however, mean that the highest AUC will not score highly on our wAUC metric - this is intended.

The following is an example of a wAUC from [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis/overview/description)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1951250%2Ff250ff6a4e04bac332fa14d539ed813e%2FKaggle.png?generation=1588207999884987&alt=media)

This example is biased with the same weights for low false positivity. 
However, the example still stands for false negatives. 
Submission 2 has the lowest AUC, but the highest wAUC since it minimizes FPR.

## What's next for metaskin
First of all, in terms of the app, we would love to make it more robust user-side with reccomendations if the lens of their camera is dirty, or the image is too bright/dark, etc.
On the business side, we would love to integrate metaskin into a REST API for other medical imaging applications to use.
For a small fee, we would present our API to be a part of a more comprehensive home diagnosis tool.

## References
1. Ha, Q., Liu, B., & Liu, F. (2020). Identifying melanoma images using efficientnet ensemble: Winning solution to the SIIM-ISIC melanoma classification challenge. ArXiv:2010.05351 [Cs]. http://arxiv.org/abs/2010.05351
2. American Cancer Society. Facts & Figures (2021). American Cancer Society. Atlanta, Ga. 2021.