# Landmark Classification and Tagging for Social Media

This project aims to classify and tag landmarks in social media images using machine learning techniques. It includes two main approaches: building a Convolutional Neural Network (CNN) from scratch and using transfer learning.

## Project Overview
The goal of this project is to develop a model that can accurately classify landmarks in social media images. The project consists of the following main components:

- **cnn_from_scratch.ipynb**: This notebook contains the code for building a CNN from scratch to classify landmarks. It includes data preprocessing, model architecture, training, and evaluation.
- **transfer_learning.ipynb**: This notebook demonstrates the use of transfer learning to classify landmarks. It uses a pre-trained model as a feature extractor and trains a linear classifier on top of it.
- **app.ipynb**: This notebook showcases the deployment of the trained model in a simple web application. It allows users to upload images and get predictions for the landmark classification.
- **src**: This directory contains the source code for the project, including helper functions, data loading, model definition, training, and prediction.

## File Requirements
To successfully run this project, make sure you have the following files in your project directory:

- cnn_from_scratch.ipynb
- transfer_learning.ipynb
- app.ipynb
- src/train.py
- src/model.py
- src/helpers.py
- src/predictor.py
- src/transfer.py
- src/optimization.py
- src/data.py


## Getting Started
To get started with this project, follow these steps:

1. Clone the project repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Open the Jupyter notebooks (cnn_from_scratch.ipynb, transfer_learning.ipynb, app.ipynb) and run the cells sequentially.

For detailed instructions and explanations, please refer to the project notebooks.


## License
This project is licensed under the MIT License.

Feel free to customize this README template according to your project's specific details and requirements. Let me know if you need any further assistance!
