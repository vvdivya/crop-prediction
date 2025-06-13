# crop-prediction
This project aims to determine the type of crop to be cultivated based on soil characteristics such as Phosphorus, Potassium, Temperature, Humidity, Soil pH, and Rainfall.
Technologies used: python, python modules: (module) pickle, numpy, sklearn, streamlit
For predicting the crop I used Machine Learning algorithm svm(support vector mechine) and used the dataset: 
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data

pickle: 
Create portable serialized representations of Python objects.
See module copyreg for a mechanism for registering custom picklers.
See module pickletools source for extensive comments.
	
 Classes:

    Pickle  
    Unpickle
Functions:

    dump(object, file)  
    dumps(object) -> string  
    load(file) -> object  
    loads(bytes) -> object


numpy: NumPy
Provides
An array object of arbitrary homogeneous items
Fast mathematical operations over arrays
Linear Algebra, Fourier Transforms, Random Number Generation

Available subpackages(This subpackages will have number of functions)
lib:Basic functions used by several sub-packages.
random:Core Random Tools
linalg:Core Linear Algebra Tools
fft:Core FFT routines
polynomial:Polynomial tools
testing:NumPy testing tools

Scikit-learn (sklearn) is a powerful and easy-to-use Python library for machine learning. It provides simple and efficient tools for data analysis and modeling, including classification, regression, clustering, and dimensionality reduction.

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that best separates different classes in the feature space, making it effective for both linear and non-linear data.

Scikit-learn provides SVM functionality through the sklearn.svm module. It includes several classes, such as:
SVC – for classification tasks (Support Vector Classification)
SVR – for regression tasks (Support Vector Regression)
LinearSVC and LinearSVR – for linear kernel SVMs (faster for large datasets)

Streamlit:is an open-source Python library used to create and share custom web apps for machine learning and data science projects. It allows you to turn data scripts into interactive dashboards with minimal code, making it ideal for quickly building user-friendly interfaces for ML models and visualizations

# Setting up development environment
	
 1. Install python(https://www.python.org/downloads/)
	2. Install skleran
 					pip install scikit-learn
 	 3.. Install streamlit(Streamlit requires Python 3.7 or higher. Check your version: python --version
						pip install streamlit
 	 4. Install Vscode(https://code.visualstudio.com/):Visual Studio Code (VS Code) is a free, lightweight, and powerful source code editor developed by Microsoft. It supports multiple programming languages, including Python, JavaScript, and C++.

# Steps to run the application
1. python svm.py ( this creates a machine learning model that can be used for predicting purpose)
   				
3. streamlit run app.py ( creates a web app so that user can input the soil parameter and get suggestion on type of crop to cultivate) 

