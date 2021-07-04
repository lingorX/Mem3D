
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/lll/liliulei'  # folder that contains VOCdevkit/.

        elif database == 'sbd':
            return '/path/to/SBD/'  # folder with img/, inst/, cls/, etc.
        elif database == 'CVC':
            return 'data' 
        elif database == 'MSD':
            return 'data' 
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'res_model_dict/resnet101-5d3b4d8f.pth'    
        #'resnet101-5d3b4d8f.pth' #resnet50-19c8e357.pth'
