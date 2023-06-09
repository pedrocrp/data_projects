import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import log_loss


class FeatureClassifier:
    def __init__(self, network):
        self.classifiers =[
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
    
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=1)),
    
    ('Random Forest', RandomForestClassifier(random_state=42, )),
    
    ('Decision Tree',DecisionTreeClassifier(random_state=42)),

    ('XGB',XGBClassifier(random_state=42)),

    ('LGBM', LGBMClassifier(random_state=42)),
]
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.network = network
        self.columns = ['Network', 'Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Kappa Score', 'Log Loss', 'Cross Val Score']
        self.df_results = pd.DataFrame(columns=self.columns)
        self.full_dir_path = os.path.dirname(os.path.realpath(__file__))
<<<<<<< HEAD
        self.features_path = fr'{self.full_dir_path}\features'
        os.makedirs(fr'{self.full_dir_path}\learning_curves', exist_ok=True)
        os.makedirs(fr'{self.full_dir_path}\confusion_matrix', exist_ok=True)
        os.makedirs(fr'{self.full_dir_path}\results', exist_ok=True)
=======
        self.features_path = fr'{self.full_dir_path}\features' 
        os.makedirs(fr'{self.full_dir_path}\learning_curves', exist_ok=True)
        os.makedirs(fr'{self.full_dir_path}\confusion_matrix', exist_ok=True)
        os.makedirs(fr'{self.full_dir_path}\results', exist_ok=True)
        
>>>>>>> 89dec17ed08c480e25506cc1e8b6022a87662c2f

        
    def load_data(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        # Replace special characters with underscore
        self.data.columns = [re.sub('[^A-Za-z0-9]+', '_', col) for col in self.data.columns]



    def split_data(self):
        x = self.data.drop("label", axis=1)
        y = self.data["label"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)


    def clean_data(self):
        paths = [fr'{self.full_dir_path}\learning_curves',
                 fr'{self.full_dir_path}\confusion_matrix',
                 fr'{self.full_dir_path}\results'
                 ]
        
        for path in paths:
            files = os.listdir(path)
            for file_name in files:
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)


    def create_label_mapping(self, main_directory):
        class_names = os.listdir(main_directory)
        label_mapping = {i: name for i, name in enumerate(class_names)}
        return label_mapping


    def fit_predict(self, model, model_name):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        y_pred_proba = model.predict_proba(self.x_test)
        self.cross_score = self.cross_val_score(model)
        self.results(model_name, self.y_test, y_pred, y_pred_proba, self.network)
        self.show_results(y_pred, self.y_test, model_name)


    def results(self, algorithm, y_test, y_pred, y_pred_proba, network):
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        kappa = cohen_kappa_score(y_test, y_pred) * 100
        logloss = log_loss(y_test, y_pred_proba)

        new_data = pd.DataFrame([{
            'Network': network,
            'Algorithm': algorithm,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Kappa Score': kappa,
            'Log Loss': logloss,
            'Cross Val Score': self.cross_score
        }])

        self.df_results = pd.concat([self.df_results, new_data], ignore_index=True)


    def show_results(self, y_test_pred, y_test, name):
        label_mapping = self.create_label_mapping(f'{self.full_dir_path}\images')
        y_test = np.array([label_mapping[label] for label in y_test])
        y_test_pred = np.array([label_mapping[label] for label in y_test_pred])

        cm = confusion_matrix(y_test, y_test_pred)

        plt.figure(figsize=(12, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix {name} - {self.network}')
        plt.savefig(fr'{self.full_dir_path}\confusion_matrix\{name}_{self.network}_confusion_matrix.png')
        plt.clf() 


    def cross_val_score(self, classifier):
        cv_scores = cross_val_score(classifier, self.x_train, self.y_train, cv=3)
        return np.mean(cv_scores) * 100


    def plot_learning_curve(self, model, model_name):
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.x_train, self.y_train, cv=5, scoring='accuracy',
            n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

        # train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, test_mean, label="Cross-validation score")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
        plt.title("Learning Curve for " + model_name)
        plt.tight_layout()
        plt.savefig(fr'{self.full_dir_path}\learning_curves\{model_name}_{self.network}_learning_curve.png')
        plt.clf() 


    def pipeline(self, csv_file_path):
        self.clean_data()
        self.load_data(csv_file_path)
        self.split_data()
        for classifier_name, classifier in self.classifiers:
            print(f'Using classifier {classifier_name} with {self.network} features')
            self.fit_predict(classifier, classifier_name)
            self.plot_learning_curve(classifier, classifier_name)
            print(f'Learning curve and Confusion Matrix finished')
        
        self.df_results = self.df_results.sort_values(by=['F1-score', 'Cross Val Score'], ascending=False)
        
        self.df_results.to_csv(fr'{self.full_dir_path}\results\{self.network}_results.csv', index=False)
        self.df_results.to_excel(fr'{self.full_dir_path}\results\{self.network}_results.xlsx', index=False)
    

    def run(self):
        for csv_file in os.listdir(self.features_path):
            if self.network in csv_file:
                csv_file_path = fr'{self.features_path}\{csv_file}'
                self.pipeline(csv_file_path)


classifier = FeatureClassifier('resnet50')
classifier.run()
            

