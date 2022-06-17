# Aim 
The aim of this learning project is two fold. The primary goal is to capture the sentiment in a movie review and hence classify the sentiment as either positive or negative. The secondary goal was is to compare two machine learning models namely logistic regression and support vector machines (SVM) in order to find out which of these models (with optimized hyperparameters) performs best.   

# Dataset
The IMDb dataset contains 50.000 different movie reviews. The reviews are either positive i.e. more than six stars or negative i.e. less than five stars. The dataset can be downloaded here: http://ai.stanford.edu/~amaas/data/sentiment/ 

In order to avoid crossing GitHub's recommended maximum file size the dataset (csv) is compressed into a zip file.

The dataset was used for the following paper: 
Maas, A. L. , Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y.,& Potts, C. (2011). 
Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)

# Methodology
In the first step html and special characters were removed from the dataset, however as emoticons convey a meaning they were kept.

Next a tokenizer function was defined with the aim of dividing each document into its fundamental parts. The function devides the documents into individual words using whitespace as seperator. 

In order to classify a sentiment expressed in text form, the wheat has to be seperated from the chaff, which is to say that the words which mostly signfy either one of the binary classes have to be distinguished from words that regularly occure in both classes. The TfidfVectorizer from scikit-learn, which basically defined is term frequency multiplied by the invrese document frequency was used for this purpose.

Two models were taken into consideration for the classification task at hand, namely logistic regression and SVM. Although both deliver similar results, logistic regresion is more vunlenrable to outliers and hence SVM was considered as a possible alternative. 

Grid Search (with a fivefold cross validation) was used for hyperparameter optimization of both the classfifier and the vectorizer. For further information regarding the list of hyperparameters used in this learning project the excellent scikit-learn documentation is highly recommended.  

# Outcomes
The first thing that stands out is that finding the best model is quite time-consuming. It took aproximately 50 minutes on my local machine. Setting n-jobs=-1 (if more than one CPU is available) might improve matters.

The best model is the logistic regression with the parameter c, as the inverse of the regularization parameter, at 10 and L2 (weight decay) as the type of regularization. The best model also happens to despense with using stop words for the vectorizer.

The accuracy score, defined as the proportion of correct classifications, of the best model when applied on the test dataset is at 90%. This means that the best model can predict a movie review as either positive or negative with an accuracy of 90%. 

# Sources
- Raschka, S. , & Mirjalili, V. (2021). Machine Learning mit Python und Keras, TensorFlow 2 und Scikit-learn: Das umfassende Praxis-Handbuch für Data Science, Deep Learning und Predictive Analytics (K. Lorenzen, Trans.) (3 edition).mitp

- Christian S. Perone, "Machine Learning :: Text feature extraction (tf-idf) – Part I," in Terra Incognita, 18/09/2011, https://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/ and https://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/

- https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers

- https://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/