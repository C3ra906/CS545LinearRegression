#CS545 Group Project. Logistic Regression

import pandas as pd
import numpy as np
from pandas.core.base import DataError
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn as sk
import random

#Data Path
path = './healthcare-dataset-stroke-data.csv'

#Function that loads and preprocesses csv data
def preprocess_data(file_path, choice):
   data = pd.read_csv(file_path)

   #Replace categorical values with numerical
   data['gender'].replace(['Male', 'Female', 'Other'], [0,1,2], inplace = True)
   data['ever_married'].replace(['No', 'Yes'], [0,1], inplace = True)
   data['work_type'].replace(['Private', 'Self-employed','Govt_job', 'children', 'Never_worked'],[0,1,2,3,4], inplace = True)
   data['Residence_type'].replace(['Rural','Urban'], [0,1], inplace = True)
   data['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'],[0,1,2,3], inplace = True)

   #remove rows with nans
   data.dropna(inplace = True)
    
   #split data by labels
   stroke = data.loc[data['stroke'] == 1]
   non_stroke = data.loc[data['stroke'] == 0]
   
   if (choice == 0):
      #Split data into even halves
      #create random subset of the row indices
      random.seed(34)
      stroke_t_rows = random.sample(range(stroke.shape[0]), k = int(stroke.shape[0]/2))
      non_stroke_t_rows = random.sample(range(non_stroke.shape[0]), k = int(non_stroke.shape[0]/2))
      #subset stroke testing and training
      stroke_t = stroke.iloc[stroke_t_rows,:].copy()
      stroke_tr = stroke.drop(stroke.index[stroke_t_rows])
      #subset non stroke testing and training
      non_stroke_t = non_stroke.iloc[non_stroke_t_rows,:].copy()
      non_stroke_tr = non_stroke.drop(non_stroke.index[non_stroke_t_rows]) 
      #concat stroke and non stroke subsets together
      t_frames = [stroke_t, non_stroke_t]
      tr_frames = [stroke_tr, non_stroke_tr]
      test_set = pd.concat(t_frames)
      train_set = pd.concat(tr_frames)

   #Use undersampling of nonstroke data
   elif (choice == 1):
      train_set, test_set = undersample(stroke, non_stroke)

   #Use oversampling of stroke data
   elif (choice == 2):
      train_set, test_set = oversample(stroke, non_stroke)

   #Use oversampling of stroke data and undersampling of nonstroke data
   else:
      train_set, test_set = over_undersample(stroke, non_stroke)

   # remove id column
   train_set = train_set.drop(columns = 'id') 
   test_set = test_set.drop(columns = 'id')

   #Scale data to range 0 to 1
   min_max_scaler = preprocessing.MinMaxScaler()
   train_set[['gender','age', 'work_type', 'smoking_status', 'avg_glucose_level', 'bmi', 'smoking_status']] = min_max_scaler.fit_transform(train_set[['gender','age', 'work_type', 'smoking_status', 'avg_glucose_level', 'bmi', 'smoking_status']])
   test_set[['gender','age', 'work_type', 'smoking_status', 'avg_glucose_level', 'bmi', 'smoking_status']] = min_max_scaler.fit_transform(test_set[['gender','age', 'work_type', 'smoking_status', 'avg_glucose_level', 'bmi', 'smoking_status']])

   #Save labels as its own vector
   test_labels = np.copy(test_set['stroke'])
   test_set['stroke'].replace([0], [1], inplace = True) #Replacing the entire row with 1s for weight x input multiplication
   train_labels = np.copy(train_set['stroke'])
   train_set['stroke'].replace([0], [1], inplace = True) #Replacing the entire row with 1s for weight x input multiplication

   return train_set, test_set, train_labels, test_labels

#Undersample Non-stroke Data
def undersample(stroke, non_stroke):
   #Undersample non stroke data by about half
   random.seed(34)
   undersample_non_stroke = non_stroke.sample(n = 2000) #picks 2000 random data points from the data set 

   #Split data into even halves
   stroke_t_rows = random.sample(range(stroke.shape[0]), k = int(stroke.shape[0]/2))
   non_stroke_t_rows = random.sample(range(undersample_non_stroke.shape[0]), k = int(undersample_non_stroke.shape[0]/2)) 
   stroke_t = stroke.iloc[stroke_t_rows,:].copy()
   stroke_tr = stroke.drop(stroke.index[stroke_t_rows])
   u_non_stroke_t = undersample_non_stroke.iloc[non_stroke_t_rows,:].copy()
   u_non_stroke_tr = undersample_non_stroke.drop(undersample_non_stroke.index[non_stroke_t_rows])

   #Merge to create test_set and train_set
   t_frames = [stroke_t, u_non_stroke_t]
   tr_frames = [stroke_tr, u_non_stroke_tr]
   test_set = pd.concat(t_frames)
   train_set = pd.concat(tr_frames)

   return train_set, test_set

#Oversample Stroke Data
def oversample(stroke, non_stroke):
   #Oversample stroke data by about half
   random.seed(34)
   oversample_stroke = stroke.sample(n = 100)
   s_frame = [oversample_stroke, stroke]
   o_stroke = pd.concat(s_frame)

   #Split data into even halves
   stroke_t_rows = random.sample(range(o_stroke.shape[0]), k = int(o_stroke.shape[0]/2))
   non_stroke_t_rows = random.sample(range(non_stroke.shape[0]), k = int(non_stroke.shape[0]/2)) 
   stroke_tr = o_stroke.iloc[stroke_t_rows,:].copy()
   stroke_t = o_stroke.drop(o_stroke.index[stroke_t_rows]) 
   non_stroke_t = non_stroke.iloc[non_stroke_t_rows,:].copy() 
   non_stroke_tr = non_stroke.drop(non_stroke.index[non_stroke_t_rows])

   #Merge to create test_set and train_set
   t_frames = [stroke_t, non_stroke_t]
   tr_frames = [stroke_tr, non_stroke_tr]
   test_set = pd.concat(t_frames)
   train_set = pd.concat(tr_frames)

   return train_set, test_set

#Oversample Non-Stroke Data and Stroke Data
def over_undersample(stroke, non_stroke):
   #Oversample stroke data by about half
   random.seed(34)
   oversample_stroke = stroke.sample(n = 100)
   s_frame = [oversample_stroke, stroke]
   o_stroke = pd.concat(s_frame)
   
   undersample_non_stroke = non_stroke.sample(n = 2000) #picks 2000 random data points from the data set 

   #Split data into even halves
   stroke_t_rows = random.sample(range(o_stroke.shape[0]), k = int(o_stroke.shape[0]/2))
   non_stroke_t_rows = random.sample(range(undersample_non_stroke.shape[0]), k = int(undersample_non_stroke.shape[0]/2)) 
   stroke_tr = o_stroke.iloc[stroke_t_rows,:].copy()
   stroke_t = o_stroke.drop(o_stroke.index[stroke_t_rows]) 
   non_stroke_t = undersample_non_stroke.iloc[non_stroke_t_rows,:].copy()
   non_stroke_tr = undersample_non_stroke.drop(undersample_non_stroke.index[non_stroke_t_rows])

   #Merge to create test_set and train_set
   t_frames = [stroke_t, non_stroke_t]
   tr_frames = [stroke_tr, non_stroke_tr]
   test_set = pd.concat(t_frames)
   train_set = pd.concat(tr_frames)

   return train_set, test_set

#Function that initializes the weights for the input x weight matrix and change matrix to keep track of changes in weights
def initialize_weights():
    #We have 10 features + b_0 
    np.random.seed(34)
    w = np.random.randint(-5, 5, size=(1, 10))/100 #set weights to random number between -0.05 to 0.05
    b_0 = np.ones((1, 1))

    #Add b_0 to the end of weight matrix since we changed the stroke column to 1s
    weights = np.append(w, b_0, axis = 1) 
    w_changes = np.zeros((1, 11))

    return weights, w_changes

#Logistic Function for Sigma Calculation
def logistic(weights, inputs):

   sigma = 1/(1 + np.exp(-1 * np.dot(weights, inputs.T)))

   return sigma

#Maximum Likelihood Estimate for Parameters
def MLE(data, labels, weights, w_changes):
    learn = 0.001 #learning rate
    sigma = logistic(weights, data) #gives matrix of sigma values
    size = np.size(labels, 0)
    labels = np.reshape(labels, (size, 1))
    runs = 0

    #Round to 4th decimial place for stopping condition check
    w_changes = np.round(w_changes, 4)
    weights = np.round(weights, 4)

    comparison = w_changes == weights

    while (comparison.all() == False): 
        w_changes = np.copy(weights)
        weights += learn * np.dot((labels.T - sigma), data) 
        sigma = logistic(weights, data)

        #Round to 4th decimial place for stopping condition check
        w_changes = np.round(w_changes, 4)
        weights = np.round(weights, 4)
        
        comparison = w_changes == weights
        runs += 1
    
    return weights

#Predicts Results, Calculates Accuracy, Prints Confusion Matrix
def predict(data, weights, labels):
    confusion = np.zeros((2,2)) #Confusion matrix [TP, FN][FP, TN]
    index = 0
    results = np.dot(data, weights.T)
    count_s = 0
    for row in results:
        if row > 0:
            if labels[index] == 1:
               count_s += 1
               confusion[0,0] += 1
            else:
               confusion[1,0] += 1               
        else:
            if labels[index] == 0:
               confusion[1,1] += 1
            else:
               count_s += 1
               confusion[0,1] += 1
        index+= 1

    tp = confusion[0,0]
    fp = confusion[1,0]
    tn = confusion[1,1]
    fn = confusion[0,1]
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    print(f"confusion matrix: \n {confusion}")
    return accuracy

#Runs Logistic Regression
def run_epoch(train_set, test_set, train_labels, test_labels): 
    epoch = 0
    weights, w_changes = initialize_weights()

    #Run Training
    while (epoch < 1):#set to 1 since weights no longer change after the first epoch
        print(f"Training until weights no longer change:")
        final_weights = MLE(train_set, train_labels, weights,w_changes)
        tr_accuracy = predict(train_set, final_weights, train_labels)
        print(f"Training accuracy: {tr_accuracy}")
        weights = np.copy(final_weights)
        epoch += 1

    #Run Test
    print(f"Testing:")
    t_accuracy = predict(test_set, final_weights, test_labels)
    print(f"Test accuracy: {t_accuracy}")


def main():
    print("Unchanged data size: ")
    train_set, test_set, train_labels, test_labels = preprocess_data(path, 0)
    run_epoch(train_set, test_set, train_labels, test_labels)
    print("\nUndersampled:")
    train_set, test_set, train_labels, test_labels = preprocess_data(path, 1)
    run_epoch(train_set, test_set, train_labels, test_labels)
    print("\nOversampled:")
    train_set, test_set, train_labels, test_labels = preprocess_data(path, 2)
    run_epoch(train_set, test_set, train_labels, test_labels)
    print("\nOver and Undersampled:")
    train_set, test_set, train_labels, test_labels = preprocess_data(path, 3)
    run_epoch(train_set, test_set, train_labels, test_labels)

if __name__ == '__main__':
    main()