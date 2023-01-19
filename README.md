# Respiratory-Sound-Classfication-Using-CNN
Model to classify respiratory diseases using respiratory sound

## Overview
In this Project I used CNN classifier For Respiratory sound Classfication to detect the diseases .

## Dataset
Respiratory sounds are important indicators of respiratory health and respiratory disorders. The sound emitted when a person breathes is directly related to air movement, changes within lung tissue and the position of secretions within the lung. A wheezing sound, for example, is a common sign that a patient has an obstructive airway disease like asthma or chronic obstructive pulmonary disease (COPD).

These sounds can be recorded using digital stethoscopes and other recording techniques. This digital data opens up the possibility of using machine learning to automatically diagnose respiratory disorders like asthma, pneumonia and bronchiolitis, to name a few.

The Respiratory Sound Database was created by two research teams in Portugal and Greece. It includes 920 annotated recordings of varying length - 10s to 90s. These recordings were taken from 126 patients. There are a total of 5.5 hours of recordings containing 6898 respiratory cycles - 1864 contain crackles, 886 contain wheezes and 506 contain both crackles and wheezes. The data includes both clean respiratory sounds as well as noisy recordings that simulate real life conditions. The patients span all age groups - children, adults and the elderly.

link :- https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
## Results

                  precision    recall  f1-score   support

             COPD       0.98      0.99      0.98       165
          Healthy       0.50      0.57      0.53         7
             URTI       0.33      0.25      0.29         4
    Bronchiectasis      1.00      0.50      0.67         4
        Pneumoina       0.60      1.00      0.75         3
    Bronchiolitis       0.00      0.00      0.00         1

        accuracy                            0.94       184
        macro avg       0.57      0.55      0.54       184
     weighted avg       0.94      0.94      0.94       184
