



class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'MTR' # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = 'data/train/0/00' # 训练集存放路径
    map_root = 'data/maps/0_cologne_butzweiler.osm'
    # test_data_root = './data/test' # 测试集存放路径
    load_model_path = None # 加载预训练的模型的路径，为None代表不加载

    batch_size = 20 # batch size
    max_epoch = 100
    
    use_gpu = False # use GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    # agent
    num_agents = 4

    # map feature
    num_points_polyline = 10
    num_src_polyline = 20

    result_file = 'result.csv'

    lr = 0.01 # initial learning rate
    lr_decay = 0.97 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

