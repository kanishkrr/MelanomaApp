# MelanomaApp

used a CNN to distinguish benign vs. malignant moles of Melanoma, one of the most serious types of skin cancer

was able to achieve 91.6% accuracy on the third attempt (80% first, 86.8% second)

techniques used: Convolution, Max Pooling, Batch Normalization, and Dropout

![image](https://github.com/kanishkrr/MelanomaApp/assets/138060333/2c0f221a-d677-49ec-8791-ba184e54b86a)

developed a web app using html, css, js (frontend) and flask (backend)

i wasn't able to upload the pytorch model since it was too large (138.9mb) - over 100mb file limit

things to improve in the future:

- scale the images down to less than 75x75 pixels since the model took over 2 hours to train (trained at 200x200 px size)
- potentially use Adam optimizer instead of Stochastic Gradient Descent to more efficiently train the model
- use a less complex Neural Network (i don't think it was necessary for such a large net - 9216 --> 4000 --> 2000 --> 2)

learned a lot with this project

dataset used: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

