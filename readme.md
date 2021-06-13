## Who we are
Hi! I'm Stanley, SDE Intern at Kings Distributed Systems/Distributed Computer.
I mainly work on ML tasks, and spend my free time on Kaggle - Google's data science competitions platform, where I am ranked 1100th globally.

## Inspiration
We were inspired by an app: [skinvision.com](https://skinvision.com). 
They charge $10 per scan of lesions. 
While we don't know what algorithms they are using, we can read about the academic State of the Art through papers.
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


## How we built it

## Challenges we ran into

## Accomplishments that we're proud of

## What we learned

## What's next for metaskin (or something like that)
We would love to make it more robust user-side with reccomendations if the lens of their camera is dirty, or the image is too bright/dark, etc.

## References
1. Ha, Q., Liu, B., & Liu, F. (2020). Identifying melanoma images using efficientnet ensemble: Winning solution to the siim-isic melanoma classification challenge. ArXiv:2010.05351 [Cs]. http://arxiv.org/abs/2010.05351
2. American Cancer Society. Facts & Figures (2021). American Cancer Society. Atlanta, Ga. 2021.