from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
# from utils import Visualizer
from config import *

# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    test_loader = DataLoader(Dataset(args, test_mode=True, is_normal=False),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    checkpoint = torch.load(args.testing_model)

    model = Model(args.feature_size, args.batch_size)
    model.load_state_dict(checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    # output_path = ''   # put your own path here
    auc = test(test_loader, model, args, device)
    #######################################
    ######PART CALCUTE ANOMOLY AUC






