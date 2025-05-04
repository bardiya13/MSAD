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
    test_loader = DataLoader(Dataset(args, test_mode=True),
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

    # Get predictions and labels from the test loader again for anomaly AUC calculation

    model.eval()
    with torch.no_grad():
        pred = torch.zeros(0, device=device)
        labels = torch.zeros(0, device=device)

        anomaly_test_loader = DataLoader(Dataset(args, test_mode=False),
                                         batch_size=1, shuffle=False,
                                         num_workers=0, pin_memory=False)

        for i, (data, label) in enumerate(anomaly_test_loader:
            inputs = data.to(device)
            score = model(inputs)
            pred = torch.cat((pred, score))
            labels = torch.cat((labels, label.to(device)))

        # Convert to numpy for AUC calculation
        labels_np = labels.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # Filter only the abnormal videos (label=1)
        abnormal_indices = labels_np == 1
        if abnormal_indices.any():
            anomaly_pred = pred_np[abnormal_indices]
            anomaly_labels = labels_np[abnormal_indices]
            # All labels are 1, so we're measuring how well the model ranks the anomalies
            anomaly_AUC = roc_auc_score(anomaly_labels, anomaly_pred)
            print(f"Anomaly_AUC (only on abnormal videos): {anomaly_AUC:.4f}")
        else:
            print("No abnormal videos found in the test set")




