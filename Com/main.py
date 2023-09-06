import argparse
from train import train_model
from evaluation import evaluation_model
from preprocess import preprocess_data
import torch

def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-project', type=str, default='openstack', help='name of the dataset')

    # Training our model
    parser.add_argument('-train', action='store_true', help='training attention model')

    parser.add_argument('-train_data', type=str, default='./data/jit/openstack_train.pkl', help='the directory of our training data')
    parser.add_argument('-test_data', type=str, default='./data/jit/openstack_test.pkl', help='the directory of our testing data')
    parser.add_argument('-dictionary_data', type=str, default='./data/jit/openstack_dict.pkl', help='the directory of our dicitonary data')

    parser.add_argument('-features', type=str, default='./data/jit/openstack_train.pkl', help='the directory of our training data features')
    parser.add_argument('-num_feature', type=int, default=14, help='number of features')
    parser.add_argument('-only', type=bool, default=False)
    parser.add_argument('-drop', type=str, default='')
    parser.add_argument('-jitbot', type=bool, default=False)

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='extracting features')
    parser.add_argument('-predict_data', type=str, help='the directory of our extracting data')
    parser.add_argument('-name', type=str, help='name of our output file')

    # Predicting our data
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')

    # Predicting our data
    parser.add_argument('--predict', action='store_true', help='predicting testing data')

    # Number of parameters for Attention model
    parser.add_argument('-embed_size', type=int, default=768, help='the dimension of embedding vector')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-hidden_size', type=int, default=32, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='model', help='where to save the snapshot')    

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import torch._dynamo as dynamo
    # torch._dynamo.config.suppress_errors = True
    # torch.backends.cudnn.benchmark = True

    if params.train is True:

        data = preprocess_data(params)

        train_model(data=data, params=params)

        print("Done")

        exit()
    
    elif params.predict is True:

        params.batch_size = 1

        data = preprocess_data(params)
        
        evaluation_model(data=data, params=params)

        print("Done")
        
        exit()