

""" Rubikloud take home problem """
import luigi
import csv
import sklearn
import pandas as pd
import collections
import os
import sys
import numpy as np
import scipy as sp
from scipy.io import loadmat,savemat
from keras import utils
import io,math
from sklearn.cross_validation import train_test_split
from operator import itemgetter

from sklearn import svm
from sklearn.svm import SVC

import pickle

# instantiate a SVM model for predictions
clf_doorprob=SVC(probability=True,kernel='sigmoid')

        
class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='clean_data.csv')
    tweet_data_dict = collections.defaultdict(list)

    # TODO...
    # Load the file contents into memeory
        
    def run(self):    

        #clean ipnut twitter data and write to output file:
        with open(self.tweet_file, 'rU', encoding="ISO-8859-1") as datafile, open(self.output_file, 'w', encoding="ISO-8859-1") as write_file:
            csvreader = csv.reader(datafile)
            csvwriter = csv.writer(write_file)
                
            ## skip header of .csv file to be read
            next(csvreader,None)
            for record in csvreader:              
    
                if record[15][:] == '' or record[15][:] == '[0.0, 0.0]':
                    pass
                else:
                    csvwriter.writerow(record)
    
        datafile.close()
        write_file.close()
        
    #check for completeness of the task(s) by checking that the output file is non empty       
    def complete(self):
        
        if((os.path.getsize(self.output_file) > 250)):
            return True
        else:
            return False        
        
        
class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    def requires(self):
        return CleanDataTask()      
    
    #tweet_file = luigi.Parameter(default='airline_tweets.csv')
    cities_file = luigi.Parameter(default='cities.csv')
    clean_file = luigi.Parameter(default='clean_data.csv')
    output_file = luigi.Parameter(default='features.csv')
    
    # TODO...

    def run(self): 
        
        city_data_dict = collections.defaultdict(list)
        uniq_city_data_dict = collections.defaultdict(list)   
        
        # euclidian distance calc. to get min inter-city dist.
        def euclid(coords_1,coords_2):
    
            dist=math.sqrt(math.pow(math.fabs(coords_2[0]-coords_1[0]),2) + math.pow(math.fabs(coords_2[1]-coords_1[1]),2))                
            return dist
        
        
        
        with io.open( self.cities_file, 'r', encoding="ISO-8859-1") as datafile:
            csvreader = csv.reader(datafile, delimiter=',')
            
            next(csvreader,None)
            for record in csvreader:              
    
                if record[1][:] == '' :
                    pass
                else:
                    city_data_dict[record[0][:]]=record[1:]
                    uniq_city_data_dict[record[1][:]]=record[1:]
            
        sentiments=[]   
        with io.open(self.clean_file, 'rU', encoding="ISO-8859-1") as file:
            
            csvreader = csv.reader(file)
            
            closest_city=[]
            uniq_city_coords=np.zeros((len(uniq_city_data_dict),2)) 
            
            #assighn labels for sentiments of tweets...
            for stuff in csvreader:   
                if not stuff:
                    continue
                emotion=str(stuff[5]).strip()
                if emotion.find("negative") != -1:
                    sentiments.append(0)   
                elif emotion.find("neutral") != -1:
                    sentiments.append(1)
                else:
                    sentiments.append(2)
                
                y=sentiments    
                    
                min_dist=float("inf")  
                runing_coords_=[]
                runing_coords_.append(float(stuff[15].split(',')[0][1:]))
                runing_coords_.append(float(stuff[15].split(',')[-1][:-1]))            
                
                #first get list of unique cities only...
                all_coord_pairs=[]
                for keys in uniq_city_data_dict.keys():
                                                                    
                    all_coord_pairs.append(float(uniq_city_data_dict[keys][3]))
                    all_coord_pairs.append(float(uniq_city_data_dict[keys][4]))
                    
                    distance=euclid(runing_coords_,all_coord_pairs)
                    
                    ##find minimum city distances:
                    all_coord_pairs=[]
                    if distance < min_dist:
                        min_dist=distance
                        min_city_key=keys
                        
                closest_city.append(min_city_key)       
        
        cpy_closest_citiesa=closest_city.copy()
        cpy_closest_citiesb=closest_city.copy()
        comonlbls=np.zeros(len(closest_city))
        
       
        # group same cities under common integer values-= toward classifiction -> one hot encoding right after loop:
        for indx,val in enumerate(cpy_closest_citiesa):  
            if isinstance(val, int):
                continue
            for indx2,val2 in enumerate(cpy_closest_citiesb):
                 
                if val == val2:
                    comonlbls[indx2]=indx    
                    cpy_closest_citiesa[indx2]=indx 
        
        X=utils.to_categorical(comonlbls)
        
        unique_cat_cityMap={}
        ## fill a dictionary with the cities as keys and the categorical values as a dictionary:
        #for ease of correct lookup in final task...
        for k,categor_city in enumerate(cpy_closest_citiesb):
            unique_cat_cityMap[categor_city]=X[k,:]
            
        city_category_map={}
        
        city_category_map['unique_city_mapping']=unique_cat_cityMap
                   
        sp.io.savemat(os.getcwd() + '/' + 'uniq_cat_cities.mat',unique_cat_cityMap)
        
        indx_1=0
        with  open(self.output_file, 'w', encoding="ISO-8859-1") as write_file:
            for rows in range(X.shape[0]):
            
                data_n_label=np.append(X[rows],[y[indx_1]],axis=0)
                indx_1+=1
                csvwriter = csv.writer(write_file)
        
                csvwriter.writerow(data_n_label)
        
            write_file.close()    
            
    
    def complete(self):
        if((os.path.getsize(self.output_file) > 1000)):
            return True        
        else:
            return False        



class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    features_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...
    
    def requires(self):
        return TrainingDataTask()    
    
    #split the datset :    
    
    def run(self):  
        
        with io.open(self.features_file, 'rU', encoding="ISO-8859-1") as file :
    
            csvreader = csv.reader(file)
            
            # for simplicity I hardcoded the feature vector and label vector dimensions 
            Y=np.zeros((855))
            X=np.zeros((855,852)) 
            
            i=0
            for rows in csvreader:   
                j=0
                split_features=str(rows)[1:-1]
                split_features_delimit=split_features.split(',')[:-1]
                for data in split_features_delimit:
                
                    #remove any white space, convert to a float and assing to X- feature matrix,
                    data=data.strip()
                    X[i,j]=float(data[1:-1])
                    j+=1
    
                Y[i]=float(str(rows[-1]).strip())
                i+=1
           
         
        clf_door = svm.SVC(probability=True)    
    
        # fit SVM model:
        svm_model=clf_door.fit(X, Y) 
     
        # open the file for writing
        fileObject = open(self.output_file,'wb') 
        
        # this writes the object a to the
        pickle.dump(svm_model,fileObject)   
        
        # here we close the fileObject
        fileObject.close()    
    
    def complete(self):
        if((os.path.getsize(self.output_file) > 1000)):
            return True        
        else:
            return False        



class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter('cities.csv')
    #city_mapping_dict= luigi.Parameter('city_mapping.mat')
    output_file = luigi.Parameter(default='scores.csv')
    
    model_file= luigi.Parameter(default='model.pkl')

    # TODO...
    
    #dependency requirements of tasks should cascade back along the pipeline from this final 'scoring' task...
    def requires(self):
        return TrainModelTask()    
    
    #split the datset :       
    def run(self):     
                
        fileObject = open(self.model_file,'r')     
        model=pickle.load( open( self.model_file ,"rb" ) )
                   
        all_results=[]
        result_str=[]
        ## was asked to use cities.csv as the test file so must harcode so as to avoid an error with the run.sh call...
        with open(os.getcwd() + '/' + 'cities.csv', 'rU', encoding="ISO-8859-1") as datafile,open(self.output_file , 'w', encoding="ISO-8859-1") as numbersFile:
            csvreader = csv.reader(datafile)
            
            next(csvreader,None)
            for record in csvreader:              
                
                city_name=record[2][:] 
            
                # look up city name's categorical mapping dict to get its category value:
                city_category_map=sp.io.loadmat(os.getcwd() + '/' + 'uniq_cat_cities.mat')
                
    
                if(city_name in city_category_map.keys()):
                          
                    try:
                        probs=model.predict_proba(city_category_map[city_name])
    
                    except:
                        
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        print('Failed at processing Image: '+ str(exc_type))
                        print('Failed at processing Image: '+ str(exc_traceback))  
    
                        continue                    
                         
                    result= [str(city_name) + str(probs[0][0]) + str(probs[0,1]) + str (probs[0,2])] 
                    result_str.append([str(city_name),str(probs[0][0]),str(probs[0,1]),str(probs[0,2])])
                    
                    
                else:
    
                    continue
                        
                
            descendingPosSentiments = sorted(result_str, key=itemgetter(3),reverse=True)    
            
            for emotions in descendingPosSentiments:
                
                resultStr= str(emotions[0]) +"," +str(emotions[1])+","+str(emotions[2]) + "," + str(emotions[3]) + "\n";
             
                numbersFile.write(resultStr)
                numbersFile.flush()
                
        
        datafile.close()
        numbersFile.close()
        
        
    #check for completeness of the task(s) by checking that the output file is non empty    
    def complete(self):
        if(os.path.getsize(self.output_file ) > 2000): 
            return True   
        else:
            return False


if __name__ == "__main__":
    luigi.run()
