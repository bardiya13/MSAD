from torch.utils.data import DataLoader
from MGFN import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm

args = option.parse_args()
from config import *
from models.mgfn import mgfn as Model
from dataset import Dataset


def test(dataloader, model, args, device):
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        featurelen = []
        for i, inputs in tqdm(enumerate(dataloader)):

            input = inputs[0].to(device)


            if len(input.size()) == 4:
                input = input.permute(0, 2, 1, 3)


            _, _, _, _, logits = model(input)


            logits = torch.squeeze(logits, 1)


            logits = torch.mean(logits, 0)


            sig = logits

            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))



        gt = np.load(args.gt, allow_pickle=True)


        pred = list(pred.cpu().detach().numpy())


        pred = np.repeat(np.array(pred), 16)


        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        return rec_auc, pr_auc


if __name__ == '__main__':
    args = option.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    model = Model()
    print(f"Model created with structure: {model}")

    shangatic = False
    if args.datasetname == "SH":
        shangatic = True

    test_loader = DataLoader(Dataset(args, test_mode=True, shangatic=shangatic),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)


    model = model.to(device)
    print("Loading model weights...")
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.testing_model).items()})
    print("Model loaded successfully")

    auc = test(test_loader, model, args, device)
    print(f"Final AUC results: {auc}")

# git config --global user.name "Bardia Soltan"
# git config --global user.email "bardisoltan@gmail.com"
# ssh-add ~/.ssh/id_ed25519