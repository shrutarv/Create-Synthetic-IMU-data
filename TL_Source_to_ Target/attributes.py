'''
Created on Aug 14, 2019

@author: fmoya
'''


import numpy as np
import logging
import pickle

class Attributes(object):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        
        self.config = config
        
        return
    
    
    ##################################################    
    #############  creates population  ###############
    ##################################################
    def creating_init_population(self):

        '''
        Returns a matrix of size [num_classes, num_attributes]
        where for each class, a random representation will be assigned
    
        '''
        
        logging.info('            Attributes: Creating population')
  
        
        attrs = np.zeros((self.config['num_classes'], self.config['num_attributes']))
        attrs = attrs.astype(int)
        attr = np.zeros((self.config['num_attributes']))
        attr = attr.astype(int)
        
        attr[0:int(self.config['num_attributes'] * 1/2)] = 1
        
        for c in range(self.config['num_classes']):
            bool_ctrl = False
            attr_shuffle = attr
            while True:
                np.random.shuffle(attr_shuffle)
                for cl in range(self.config['num_classes']):                
                    if np.sum(attr_shuffle == attrs[cl]) == self.config['num_attributes']:
                        bool_ctrl = True
                        break
                
                if bool_ctrl == False:
                    break
            attrs[c,:] = attr_shuffle
           
        logging.info('            Attributes: Initial attributes \n{}'.format(attrs))   

        return attrs
    
    
    
    
    
    ##################################################    
    #############  Local Mutation  ###############
    ##################################################
    def mutation_local(self, attrs):

        logging.info('            Attributes: Local Mutation attrs')
        
        for idx in range(attrs.shape[0]):
            while True:
                k = np.random.randint(low = 0, high= self.config['num_attributes'])
                attrs_new = attrs[idx]
                attrs_new[k] = 1 - attrs_new[k]
                
                bool_ctrl = False
                for idy in range(idx):
                    if np.sum(attrs_new == attrs[idy]) == self.config['num_attributes']:
                        bool_ctrl=True
                        break
                        
                if np.sum(attrs_new) >= self.config['num_attributes'] // 4  and np.sum(attrs_new) < int(self.config['num_attributes'] * 7 / 8) and bool_ctrl == False:
                    logging.info('            Attributes: Mutating attr {} in pos {} \n{}\n'.format(idx, k, attrs_new))
                    attrs[idx] = attrs_new
                    break

            
        return attrs.astype(int)
    
    
    
    
    
    ##################################################    
    #############  NonLocal Mutation  ###############
    ##################################################
    def mutation_nonlocal(self, attrs, number_K = 2):

        logging.info('            Attributes: NonLocal Mutation attrs')
        
        attrs_new = np.copy(attrs)
        
        for idx in range(attrs.shape[0]):
            
            K = []
            for k_idx in range(number_K):
                while True:
                    k = np.random.randint(low = 0, high= self.config['num_attributes'])
                    
                    if k in K:
                        continue
                    attr_new = attrs[idx]
                    attr_new[k] = 1 - attr_new[k]
                    
                    bool_ctrl = False
                    for idy in range(idx):
                        if np.sum(attr_new == attrs[idy]) == self.config['num_attributes']:
                            bool_ctrl=True
                            break
                            
                    if np.sum(attr_new) >= self.config['num_attributes'] // 4  and np.sum(attr_new) < int(self.config['num_attributes'] * 3 / 4) and bool_ctrl == False:
                        logging.info('            Attributes: Mutating attr {} in pos {} \n{}\n'.format(idx, k, attr_new))
                        attrs_new[idx,:] = attr_new
                        K.append(k)
                        break
        
        
        return attrs_new.astype(int)
    
    
    
    
    ##################################################    
    #######  Non Local Mutation percentage  ##########
    ##################################################
    def mutation_nonlocal_percentage(self, attrs, percentage_pred, number_K = 6):

        logging.info('            Attributes: NonLocal Mutation attrs')

        
        attrs_new = np.copy(attrs)
        
        for idx in range(attrs.shape[0]):
            k_mutations = np.int(np.round( (1 - percentage_pred[idx]) * number_K)) + 1
            
            logging.info('            Attributes: Mutating attr {} with {} mutations \n'.format(idx, k_mutations))
            
            K = []
            for k_idx in range(k_mutations):
                while True:
                    k = np.random.randint(low = 0, high= self.config['num_attributes'])
                    
                    if k in K:
                        continue
                    attr_new = attrs[idx]
                    attr_new[k] = 1 - attr_new[k]
                    
                    bool_ctrl = False
                    for idy in range(idx):
                        if np.sum(attr_new == attrs[idy]) == self.config['num_attributes']:
                            bool_ctrl=True
                            break
                            
                    if np.sum(attr_new) >= self.config['num_attributes'] // 4  and np.sum(attr_new) < int(self.config['num_attributes'] * 3 / 4) and bool_ctrl == False:
                        logging.info('            Attributes: Mutating attr {} in pos {} \n{}\n'.format(idx, k, attr_new))
                        attrs_new[idx,:] = attr_new
                        K.append(k)
                        break
        
        
        return attrs_new.astype(int)
    
    
    

    ##################################################    
    #############  Global Mutation  ###############
    ##################################################
    def mutation_global(self, attrs):
        logging.info('            Attributes: Global Mutation attrs')
        
        new_attrs = np.zeros(attrs.shape)
        for at in range(self.config['num_classes']):
            while True:
                for a in range(self.config['num_attributes']):
                    flip_a = np.random.randint(2)
                    new_attrs[at, a] = abs(attrs[at, a] - flip_a)
                    
                bool_ctrl = False
                for idy in range(at):
                    if np.sum(new_attrs[at] == attrs[idy]) == self.config['num_attributes']:
                        bool_ctrl=True
                        break
                
                if bool_ctrl == False:
                    break
            
            logging.info('            Attributes: Global Mutating attr {} \n{}\n'.format(at, new_attrs[at]))
            
        
        return new_attrs
    
    


    ##################################################    
    #############  Adding attributes  ###############
    ##################################################
    def adding_attributes(self, attrs, k_new = 2):
        add_attrs = np.random.randint(2, size = (self.config['num_classes'], k_new))
        new_attrs = np.concatenate((attrs, add_attrs), axis = 1)
        
        self.config['num_attributes'] = self.config['num_attributes'] + k_new
        
        return new_attrs
    
    
    
    ##################################################    
    #############  Saving attributes  ###############
    ##################################################
    def save_attrs(self, attrs, fitness, itera, name_file = 'attrs', protocol_file = 'wb'):
    
        '''
        Save the attribute representation (matrix) in a pkl file.
        
        @param attrs: Matrix of attributes
        @param fitness: Accuracy or F1 metric for the evolution
        @param itera: Iteration of the evolution
        @param name_file: Name of the file
        @param protocol_file: Writing protocol: w for writing new file and b for binary
    
        '''

        logging.info('            Attributes: Saving the weights \n')
        attrs_dict = {"itera" : itera, 'attrs' : attrs, 'fitness' : fitness}
        f = open(self.config['folder_exp'] + name_file +'.pkl', protocol_file)
        pickle.dump(attrs_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        
        return
    
    

    ##################################################    
    #############  Loading attributes  ###############
    ##################################################
    def load_attrs(self, name_file = 'attrs'):

        '''
        Load the attribute representations (matrix) from a pkl file.
        
        @param name_file: Name of the file
    
        '''
        

        logging.info('            Attributes: Loading attrs')
        
        
        attrs = []
        with open('../' + self.config['folder_exp'] + '/' + name_file +'.pkl', "rb") as f:
            while True:
                try:
                    attr = pickle.load(f)
                    attrs.append(attr)
                except EOFError:
                    break
        

        logging.info('            Attributes: Number of attrs {}'.format(len(attrs)))
        
        return attrs



    