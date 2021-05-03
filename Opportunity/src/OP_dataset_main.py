'''
Created on Jun 19, 2019

@author: fmoya
'''


from OrderPicking_Dataset import Dataset


if __name__ == '__main__':
    
    
    dataset = Dataset()
    train_v, train_l, test_v, test_l, class_dict  = dataset.load_data_2(wr = '_DO', test_id = 3, batch_size = 100,
                                                                            aug_data = False, train_or_test = False,
                                                                            all_labels = False)