from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L, preprocess_input as efficientnet_v2_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os
import pandas as pd
from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class FeatureExtractor:
    def __init__(self):
        self.model_type = None
        self.model = None
        self.img_size = (224, 224)  # Default for VGG16 and VGG19
        self.supported_models = {'vgg16': 4096, 'vgg19': 4096, 'resnet50': 2048, 'efficientnet_v2': 1280, 'inception_v3': 2048}
        self.count_img = 1
        self.full_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.features_path = fr'{self.full_dir_path}\features' 
        os.makedirs(fr'{self.full_dir_path}\features', exist_ok=True)


    def _init_model(self, model_type):
        self.model_type = model_type
        if model_type == 'vgg16':
            base_model = VGG16(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        elif model_type == 'vgg19':
            base_model = VGG19(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        elif model_type == 'inception_v3':
            base_model = InceptionV3(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.img_size = (299, 299)
        elif model_type == 'efficientnet_v2':
            base_model = EfficientNetV2L(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            self.img_size = (480, 480)
        elif model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        else:
            raise ValueError("Invalid model_type provided. Supported models: 'vgg16', 'vgg19', 'inception_v3', 'efficientnet_v2', 'resnet50'.")

        print(base_model)


    def extract(self, img):
        img = img.resize(self.img_size)
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if self.model_type == 'vgg16':
            x = vgg16_preprocess_input(x)
        elif self.model_type == 'vgg19':
            x = vgg19_preprocess_input(x)
        elif self.model_type == 'inception_v3':
            x = inception_v3_preprocess_input(x)
        elif self.model_type == 'efficientnet_v2':
            x = efficientnet_v2_preprocess_input(x)
        elif self.model_type == 'resnet50':
            x = preprocess_input(x)

        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)


    def extract_features_and_labels(self, directory, label, total_num_images):
        features = []
        labels = []

        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            print(f'Extracting from {img_name} || {self.count_img} images from {total_num_images} in {directory}')
            img = Image.open(img_path)
            feature = self.extract(img)
            features.append(feature)
            labels.append(label)
            self.count_img += 1

        return features, labels


    def extract_features_and_create_csv(self, class_dirs, num_columns):
        all_features = []
        all_labels = []
        total_num_images = 0

        for dir in class_dirs:
            total_num_images += len(os.listdir(dir[0]))

        print(f'----------- Extracting features with {self.model_type} -----------')
        for class_dir, label in class_dirs:
            class_features, class_labels = self.extract_features_and_labels(class_dir, label, total_num_images)
            all_features += class_features
            all_labels += class_labels
            print(f'++++++++++ Extracted features from {class_dir} ++++++++++')

        column_names = [f'column_{i}' for i in range(num_columns)]
        df = pd.DataFrame(all_features, columns=column_names)
        df['label'] = all_labels
        df.to_csv(fr'{self.features_path}\features_{self.model_type}.csv', index=False)
    

    def run(self, parent_directory, model_type):
        # List all subdirectories in the parent directory
        image_directories = [os.path.join(parent_directory, name) for name in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, name))]

        # Define labels based on directory names
        class_dirs = [(dir, i) for i, dir in enumerate(image_directories)]

        if model_type == 'all':
            for model_type, num_columns in self.supported_models.items():
                self._init_model(model_type)
                self.extract_features_and_create_csv(class_dirs, num_columns)
        else:
            self._init_model(model_type)
            num_columns = self.supported_models[self.model_type]
            self.extract_features_and_create_csv(class_dirs, num_columns)


fe = FeatureExtractor()
fe.run(r'image_processing\images', 'vgg19')
