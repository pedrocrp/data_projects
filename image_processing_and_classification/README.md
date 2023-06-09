# Image Processing and Classification
This repository showcases a project related to image processing and classification tasks using Python. The projects in this repository leverage popular libraries, popular Neural Networks and techniques to extract features from images, train machine learning models, and classify images into different categories.

## Project Overview
The projects in this repository demonstrates various aspects, including feature extraction from images using pre-trained deep learning models, building and evaluating classification models, generating learning curves, and analyzing performance metrics such as accuracy, precision, recall, and F1-score.

## Folder Structure
The repository is organized into the following structure:

 - image_processing: This folder contains images used in the image processing and classification projects.
 - learning_curves: This folder stores the learning curve plots generated during the analysis.
 - confusion_matrix: This folder contains the confusion matrix plots generated during the evaluation of classification models.
 - results: This folder stores the results of the classification models, including performance metrics and cross-validation scores.
 - features: This folder contains the extracted features from the images using different pre-trained deep learning models.

## Technologies and Libraries Used
The projects in this repository are implemented using Python and leverage various libraries and frameworks, including:

 - TensorFlow: An open-source deep learning framework used for feature extraction and building classification models.
 - Pandas: A powerful data manipulation library used for data loading, preprocessing, and analysis.
 - NumPy: A fundamental library for numerical computing in Python, used for handling multi-dimensional arrays and calculations.
 - Matplotlib and Seaborn: Visualization libraries used for creating plots, including learning curves and confusion matrices.
 - Scikit-learn: A comprehensive machine learning library used for model evaluation and performance metrics.
 - XGBoost and LightGBM: Gradient boosting frameworks used for building ensemble models.
 - OpenCV: A versatile computer vision library used for image processing and manipulation.

## How to Use the Repository
To explore the projects in this repository, follow these steps:

1. Clone the repository to your local machine using the following command:
bash
git clone <repository-url>
2. Install the required dependencies specified in the project's requirements.txt file.

3. Navigate to the specific project folder and review the project's code, notebooks, and data.

4. Execute the code or run the notebooks to reproduce the results and gain insights into image processing and classification tasks.
  
# Scripts
## FeatureExtractor
The FeatureExtractor class is responsible for extracting features from images using pre-trained deep learning models. The class supports various models such as VGG16, VGG19, ResNet50, InceptionV3, and EfficientNetV2L. The extracted features are normalized and saved as CSV files for further analysis and classification.  
 
## FeatureClassifier
The FeatureClassifier class performs image classification using different classifiers such as Logistic Regression, K-Nearest Neighbors, Random Forest, Decision Tree, XGBoost, and LightGBM. The class loads the extracted features from CSV files, splits the data into training and testing sets, fits the classifiers, evaluates the models using performance metrics, and generates learning curves and confusion matrices.
  
## Running the Code
To run the projects in this repository, follow these steps:

1. Ensure that you have the required dependencies installed.

2. Update the file paths and configurations in the code as per your system and project setup.

3. To use the Extractor, you have to create a directory named "images" with the images you want to extract and separate then in subdirs with their class names. For example, if you have images from animals, your directory may be like this:

 Main Directory
 - feature_classifier.py
 - feature_extraction.py
 - images
   - dog
   - cat
   - duck

4. Execute the code or run the notebooks to perform feature extraction, classification, and analysis.
  
5. You have to first extract the features using the FeatureExtractor class file and after this use the FeatureClassifier class to train the models.

## Results and Evaluation

The classification models' results, including accuracy, precision, recall, F1-score, cross-validation scores, and log loss, are stored in the results directory of the repository. The results are saved in both CSV and Excel formats for easy analysis and comparison.

The results directory contains individual result files for each network architecture used in the classification. Each file follows the naming convention {network}_results.csv and {network}_results.xlsx, where {network} represents the specific network architecture.

Additionally, the learning_curves directory contains learning curve plots for each classifier and network combination. These plots provide insights into the model's performance as the training data size increases.

The confusion_matrix directory contains confusion matrix plots, which visually represent the performance of the classifiers by showing the number of correct and incorrect predictions for each class.

To view and analyze the results, you can refer to the generated result files and visualization plots in the respective directories.

It's important to note that the evaluation metrics and visualizations provide valuable insights into the performance of the classification models. These results can be used to assess the effectiveness of the image processing and classification algorithms applied to the given dataset.

Feel free to explore the results and leverage them to draw conclusions and make informed decisions about the image processing and classification tasks.
  
  
  
  
  
  
  
  
  
  
  
  

