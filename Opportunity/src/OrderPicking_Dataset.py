'''
Created on Jun 19, 2019

@author: fmoya
'''



import numpy as np
import theano
#from augmentations import ActivityAugmentation

import matplotlib.pyplot as plt 

class Dataset(object):
    '''
    classdocs
    '''


    def __init__(self, ann_type = 'center'):
        '''
        Constructor
        '''
        self.path = '/vol/actrec/icpram-data/numpy_arrays/'
        self.ann_type = ann_type
        

    def get_labels(self):
        labels_dict = {0 : "NULL" , 1 : "UNKNOWN", 2 : "FLIP", 3 : "WALK",
                       4 : "SEARCH", 5 : "PICK", 6 : "SCAN", 7 : "INFO",
                       8 :"COUNT", 9: "CARRY", 10 : "ACK"}    
    
        
        return labels_dict
        
    
    
        
    def load_data(self, wr = '_DO', test_id = 3, batch_size = 1, aug_data = False):
        '''
        Loads (np arrays) per warehouse and person id
    
        Parameters:
            wr - The name of warehouse: _DO or _NP 
            test_id - id of the test person
            batch_size - batch_size
            aug_data - augmentation through interpolation, no in use
    
        Returns
            an array containing each n-dimensional window from a
        '''
        
        '''
        Loads image (np array) paths and annos
        '''
        dictz = {"_DO":{1:"004", 2:"011", 3:"017"}, "_NP":{1:"004", 2:"014", 3:"015"}}
        self.logger.info("Data: Load data dor dataset: wr {}; test person {}".format(wr, test_id))
        print "Data: Loading data dor dataset: wr {}; test person {}".format(wr, test_id)
        train_ids = (dictz[wr]).keys()
        train_ids.remove(test_id)
        train_list = ["/vol/actrec/icpram-data/numpy_arrays/%s__%s_data_labels_every-frame_100.npz"%(wr, dictz[wr][train_ids[i]]) for i in [0,1]]
        test_list= ["/vol/actrec/icpram-data/numpy_arrays/%s__%s_data_labels_every-frame_100.npz"%(wr, dictz[wr][test_id])]
        
        
        train_vals = []
        train_labels = []
        self.logger.info("Data: Load train data...")
        print "Data: Loading train data..."
        
        for path in train_list:
            tmp = np.load(path)
            vals = tmp["arr_0"].copy()
            labels = tmp["arr_1"].copy()
            tmp.close()
            
            rm_indices = []
            for i in xrange(len(labels)):
                if self.ann_type == "center":
                    # It takes the center value as label 
                    label_arg = labels[i].flatten()
                    label_arg = label_arg.astype(int)
                    label_arg = label_arg[int(label_arg.shape[0]/2)]
                else:
                    raise RuntimeError("unkown annotype")
                
                # Removing windows with label zero (NULL)
                if label_arg == 0:
                    rm_indices.append(i)
                else:
                    train_labels.append(label_arg)
                    
            self.logger.info("Data: Load train-data without 0 labels...")
            print "Data: Load train-data without 0 labels..."
            
            for i in xrange(len(labels)):
                if not i in rm_indices:
                    train_vals.append(vals[i])
                else:
                    pass
                
                
            self.logger.info("Data: Train-data without 0 labels done")
            print "Data: Train-data without 0 labels done"
            
                  
        
        
        test_vals = []
        test_labels = []
        self.logger.info("Data: Load test-data...")
        print "Data: Loading test-data..."

        tmp = np.load(test_list[0])
        vals = tmp["arr_0"].copy()
        labels = tmp["arr_1"].copy()
        tmp.close()
        
        rm_indices = []
        for i in xrange(len(labels)):
            if self.ann_type == "center":
                # It takes the center value as label 
                label_arg = labels[i].flatten()
                label_arg = label_arg.astype(int)
                label_arg = label_arg[int(label_arg.shape[0]/2)]
            else:
                raise RuntimeError("unkown annotype")
            
            # Removing windows with label zero (NULL)
            if label_arg == 0:
                rm_indices.append(i)
            else:
                test_labels.append(label_arg)
                
        self.logger.info("Data: Load-test data without 0 labels...")
        print "Data: Load-test data without 0 labels..."
        
        for i in xrange(len(labels)):
            if not i in rm_indices:
                test_vals.append(vals[i])
            else:
                pass
            
            
        self.logger.info("Data: Test-data without 0 labels done")
        print "Data: Test-data without 0 labels done"
            
            
        
        self.logger.info("Data: Loading data done")
        print "Data: Loading data done"
        
        train_v_b = np.array(self.prepare_data(np.array(train_vals), batch_size = 1))
        train_l_b = np.array(self.prepare_data(np.array(train_labels), batch_size = 1))
        test_v_b = np.array(self.prepare_data(np.array(test_vals), batch_size = 1))
        test_l_b = np.array(self.prepare_data(np.array(test_labels), batch_size = 1))
        
        
        train_v_b = self.random_data(train_v_b)
        train_l_b = self.random_data(train_l_b)
        test_v_b = self.random_data(test_v_b)
        test_l_b = self.random_data(test_l_b)
        
        return train_v_b.astype(theano.config.floatX), train_l_b.astype(theano.config.floatX), test_v_b.astype(theano.config.floatX), test_l_b.astype(theano.config.floatX)
    
    

    def load_data_2(self, wr = '_DO', test_id = 3, batch_size = 1, aug_data = False, train_or_test = False, all_labels=False):
        '''
        Loads (np arrays) per warehouse and person id
    
        Parameters:
            wr - The name of warehouse: _DO or _NP 
            test_id - id of the test person
            batch_size - batch_size
            aug_data - augmentation through interpolation, no in use
    
        Returns
            an array containing each n-dimensional window from a
        '''
        
        
        dictz = {"_DO":{1:"004", 2:"011", 3:"017"}, "_NP":{1:"004", 2:"014", 3:"015"}}
        self.logger.info("Data: Load data dor dataset: wr {}; test person {}".format(wr, test_id))
        print "Data: Loading data dor dataset: wr {}; test person {}".format(wr, test_id)
        train_ids = (dictz[wr]).keys()
        train_ids.remove(test_id)
        train_list = ["/vol/actrec/icpram-data/numpy_arrays/%s__%s_data_labels_every-frame_100.npz"%(wr, dictz[wr][train_ids[i]]) for i in [0,1]]
        test_list= ["/vol/actrec/icpram-data/numpy_arrays/%s__%s_data_labels_every-frame_100.npz"%(wr, dictz[wr][test_id])]
        
        
        train_vals = []
        train_labels = []
        self.logger.info("Data: Load train data...")
        print "Data: Loading train data..."
        
        for path in train_list:
            tmp = np.load(path)
            vals = tmp["arr_0"].copy()
            labels = tmp["arr_1"].copy()
            tmp.close()
            
            for i in xrange(len(labels)):
                train_vals.append(vals[i])
                
                
                if all_labels:
                    train_labels.append(labels[i])
                else:
                    if self.ann_type == "center":
                        # It takes the center value as label 
                        label_arg = labels[i].flatten()
                        label_arg = label_arg.astype(int)
                        label_arg = label_arg[int(label_arg.shape[0]/2)]
                    else:
                        raise RuntimeError("unkown annotype")
                    train_labels.append(label_arg)
                       
        # Make train arrays a numpy matrix
        train_vals = np.array(train_vals)
        train_labels = np.array(train_labels)  
    

        # Load the test data     
        test_vals = []
        test_labels = []
        self.logger.info("Data: Load test-data...")
        print "Data: Loading test-data..."

        tmp = np.load(test_list[0])
        vals = tmp["arr_0"].copy()
        labels = tmp["arr_1"].copy()
        tmp.close()
        
        for i in xrange(len(labels)):
            test_vals.append(vals[i])
            if all_labels:
                test_labels.append(labels[i])
            else:
                if self.ann_type == "center":
                    # It takes the center value as label 
                    label_arg = labels[i].flatten()
                    label_arg = label_arg.astype(int)
                    label_arg = label_arg[int(label_arg.shape[0]/2)]
                else:
                    raise RuntimeError("unkown annotype")
                test_labels.append(label_arg)
            
            

        # Make train arrays a numpy matrix
        test_vals = np.array(test_vals)
        test_labels = np.array(test_labels)   
            

        self.logger.info("Data: Test-data done")
        print "Data: Test-data done"
        

        
        #ch_max_min_median = np.zeros((train_vals.shape[2], 3))
        ##############################
        #Normalizing the data to be in range [0,1] following the paper
        for ch in range(train_vals.shape[2]):
            max_ch = np.max(train_vals[:,:,ch])
            min_ch = np.min(train_vals[:,:,ch])
            median_old_range = (max_ch + min_ch) / 2
            train_vals[:,:,ch] = (train_vals[:,:,ch] - median_old_range ) / (max_ch - min_ch) #+ 0.5 
            
            
        #Normalizing the data to be in range [0,1] following the paper
        for ch in range(test_vals.shape[2]):
            max_ch = np.max(test_vals[:,:,ch])
            min_ch = np.min(test_vals[:,:,ch])
            #max_ch = np.max(train_vals[:,:,ch])
            #min_ch = np.min(train_vals[:,:,ch])
            median_old_range = (max_ch + min_ch) / 2
            test_vals[:,:,ch] = (test_vals[:,:,ch] - median_old_range ) / (max_ch - min_ch) #+ 0.5 
            
        #calculate number of labels
        labels=set([])
        labels=labels.union(set(train_labels.flatten()))
        labels=labels.union(set(test_labels.flatten()))
        
        # Remove NULL class label -> should be ignored
        labels = sorted(labels)
        if labels[0] == 0:
            labels = labels[1:]
            

        #
        # Create a class dictionary and save it
        # It is a mapping from the original labels
        # to the new labels, due that the all the
        # labels dont exist in the warehouses
        #
        #
        class_dict = {}
        for i,label in enumerate(labels):
            class_dict[label]=i
        
        self.class_dict = class_dict
        

        self.logger.info("Data: class_dict {}".format(class_dict))
        print "Data: class_dict {}".format(class_dict)

        self.logger.info("Data: Augmentation of the data...")
        print "Data: Augmentation of the data..."
        

        # Print some statistics count before augmentation
        for l_i in labels:
            n_of_x_label = train_labels == l_i
            self.logger.info('{} samples for label {} before augmentation'.format( np.sum(n_of_x_label), l_i ))  
            print '{} samples for label {} before augmentation'.format( np.sum(n_of_x_label), l_i )
        
        
        if train_or_test == False:
            #
            # Create batches of train indices
            # Augment more samples for rare classes
            #
            NUM_SAMPLES = 100000
            if train_labels.shape[0] < NUM_SAMPLES:
                # First balance classes a bit nicer
                batch_train_idx, train_vals, train_labels = ActivityAugmentation.augment_by_ratio(train_vals, train_labels, labels, min_sample_ratio=0.2)
                #NUM_SAMPLES = train_labels.shape[0]
                # If neccessary augment
                #batch_train_idx, train_vals, train_labels = ActivityAugmentation.augment_by_number(train_vals, train_labels, labels, number_target_samples=NUM_SAMPLES)
            else:
                batch_train_idx, train_vals, train_labels = ActivityAugmentation.augment_by_number(train_vals, train_labels, labels, number_target_samples=1)
                
    
    
            self.logger.info("Data: Augmentation of the data done")
            print "Data: Augmentation of the data done"
            
            # Print some statistics count
            for l_i in labels:
                self.logger.info('{} samples for label {}'.format( batch_train_idx[l_i].shape[0], class_dict[l_i] ))  
                print '{} samples for label {}'.format( batch_train_idx[l_i].shape[0], class_dict[l_i] )
                           


        self.logger.info("Data: Creating final matrices with new labels and no Null label...")
        print "Data: Creating final matrices with new labels and no Null label..."
        

        counter = 0
        train_vals_fl = []
        train_labels_fl = []
        for idx in range(train_labels.shape[0]):
            if train_or_test == False:
                if counter >= NUM_SAMPLES:
                    break
            item = np.copy(train_vals[idx])
            label = train_labels[idx]
            
            if label == 0:
                continue
            train_vals_fl.append(item)
            train_labels_fl.append(int(class_dict[label]))
            
            counter += 1
        
        train_vals_fl = np.array(train_vals_fl)
        train_labels_fl = np.array(train_labels_fl)
        del train_vals
        del train_labels
        
        

        counter = 0
        test_vals_fl = []
        test_labels_fl = []
        for idx in range(test_labels.shape[0]):
            item = np.copy(test_vals[idx])
            label = test_labels[idx]
            
            if label == 0:
                continue
            test_vals_fl.append(item)
            test_labels_fl.append(int(class_dict[label]))
            
            counter += 1

        test_vals_fl = np.array(test_vals_fl)
        test_labels_fl = np.array(test_labels_fl)
        del test_vals
        del test_labels
        

        self.logger.info("Data: Randomizing the data...")
        print "Data: Randomizing the data..."
        
        
        if True:
            train_vals_fl, train_labels_fl = self.random_data(train_vals_fl, train_labels_fl)
            #test_v_b, test_l_b = self.random_data(test_v_b, test_l_b)
                
            self.logger.info("Data: Done creating final matrices with new labels and no Null label...")
            print "Data: Done creating final matrices with new labels and no Null label..."
    
            train_v_b = np.array(self.prepare_data(np.array(train_vals_fl), batch_size = batch_size))
            train_l_b = np.array(self.prepare_data(np.array(train_labels_fl), batch_size = batch_size))
            test_v_b = np.array(self.prepare_data(np.array(test_vals_fl), batch_size = 1))
            test_l_b = np.array(self.prepare_data(np.array(test_labels_fl), batch_size = 1))
    
    
            return train_v_b.astype(theano.config.floatX), train_l_b.astype(theano.config.floatX), test_v_b.astype(theano.config.floatX), test_l_b.astype(theano.config.floatX), class_dict
        else:
            train_vals_fl, train_labels_fl = self.random_data(train_vals_fl, train_labels_fl)
            #test_v_b, test_l_b = self.random_data(test_v_b, test_l_b)
                
            self.logger.info("Data: Done creating final matrices with new labels and no Null label...")
            print "Data: Done creating final matrices with new labels and no Null label..."
    
            train_v_b = np.array(self.prepare_data(np.array(train_vals_fl), batch_size = batch_size))
            train_l_b = np.array(self.prepare_data(np.array(train_labels_fl), batch_size = batch_size))
            test_v_b = np.array(self.prepare_data(np.array(test_vals_fl), batch_size = batch_size))
            test_l_b = np.array(self.prepare_data(np.array(test_labels_fl), batch_size = batch_size))
    
    
            return train_vals_fl, train_labels_fl, test_v_b.astype(theano.config.floatX), test_l_b.astype(theano.config.floatX), batch_train_idx, class_dict
    
    
        
        
    
    def prepare_data(self, data, batch_size = 1):

        self.logger.info("Prepare: Preparing data with batch size {}".format(batch_size))
        print "Prepare: Preparing data with batch size {}".format(batch_size)
        data_batches = []
        batches = np.arange(0, data.shape[0], batch_size)
        
        for idx in range(batches.shape[0] - 1 ):
            batch = []
            for data_in_batch in data[batches[idx]: batches[idx + 1]]:
                channel = []
                channel.append(data_in_batch.astype(theano.config.floatX))
                batch.append(channel)
            data_batches.append(batch)
            

        return data_batches
    
    
    def random_data(self, data, label):
        if data.shape[0] != label.shape[0]:
            self.logger.error("Random: Data and label havent the same number of samples")
            print "Random: Data and label havent the same number of samples"
            raise RuntimeError('Random: Data and label havent the same number of samples')
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        
        data_s = data[idx]
        label_s = label[idx]
        return data_s, label_s