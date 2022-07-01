# Credit_Risk_Analysis

## Overview

We are using our knowledge of Machine Learning to analyze credit card risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we needed to employ different techniques to train and evaluate models with unbalanced classes. We oversampled the data using the RandomOverSampler and SMOTE algorithms, undersampled the data using the ClusterCentroids algorithm, and used a combinatorial approach of over and undersampling using the SMOTEENN algorithm. We then compared two  machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Finally, we evaluated the performance of these models in order to make a written recommendation on whether they should be used to predict credit risk.

#### Libraries: imbalanced-learn, scikit-learn

## Results

### Oversampling
#### Naive Random Oversampling
##### Confusion Matrix

![nro_cm](https://user-images.githubusercontent.com/99751636/176283797-ac56c363-038e-4f21-884e-e41b3b95c59d.png)

##### Classification Report

![nro_cr](https://user-images.githubusercontent.com/99751636/176283804-f9536da8-ca9a-48c6-9899-deb3e459b9f3.png)

* Balanced Accuracy Score: 0.640324421824783

#### SMOTE
##### Confusion Matrix

![SMOTE_cm](https://user-images.githubusercontent.com/99751636/176283849-22f0f99c-2948-439c-9883-c45a338700ff.png)

##### Classification Report

![SMOTE_cr](https://user-images.githubusercontent.com/99751636/176283854-2b6a95f7-f20b-4b65-b9d0-9aa300eb7748.png)

* Balanced Accuracy Score: 0.6514992150524688

### Undersampling
#### Cluster Centroids
##### Confusion Matrix

![ccu_cm](https://user-images.githubusercontent.com/99751636/176283897-03f48b36-a29d-44bb-89d1-8fe34f120af1.png)

##### Classification Report

![ccu_cr](https://user-images.githubusercontent.com/99751636/176283899-dc28c746-63e4-4785-b261-1eb9516e144f.png)

* Balanced Accuracy Score: 0.5443538770387797

### Combination of over/under sampling
#### SMOTEENN
##### Confusion Matrix

![SMOTEEN_cm](https://user-images.githubusercontent.com/99751636/176283925-1d31655e-0bd3-4d35-8dde-78e6addb0408.png)

##### Classification Report

![SMOTEEN_cr](https://user-images.githubusercontent.com/99751636/176283927-77a7a1f8-0059-4b35-96ee-46fe2514fc6d.png)

* Balanced Accuracy Score: 0.6550612907408608

### Machine Learning: Ensembled Learners
#### Balanced Random Forest Classifier
##### Confusion Matrix

![brfc_cm](https://user-images.githubusercontent.com/99751636/176804761-6c599aa2-9882-4476-b40f-9de00e47aabe.png)

##### Classification Report

![brfc_cr](https://user-images.githubusercontent.com/99751636/176804779-a4769618-ada4-49fd-9a1e-8b367c5c84af.png)

* Balanced Accuracy Score: 0.5693784723481232

#### Easy Ensemble AdaBoost Classifier
##### Confusion Matrix

![eec_cm](https://user-images.githubusercontent.com/99751636/176283983-56957ec6-47de-4c0d-b6b8-8b71b5b7f6f8.png)

##### Classification Report

![eec_cr](https://user-images.githubusercontent.com/99751636/176283986-505a3575-5f24-45ab-8205-9d9b48d0c7f1.png)

* Balanced Accuracy Score: 0.4884694876536495

## Summary
