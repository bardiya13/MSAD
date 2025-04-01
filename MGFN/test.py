from torch.utils.data import DataLoader
import option
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
            print(f"\n--- Processing batch {i} ---")
            input = inputs[0].to(device)
            print(f"Input shape: {input.shape}")

            if len(input.size()) == 4:
                input = input.permute(0, 2, 1, 3)
                print(f"Shape after permute: {input.shape}")

            _, _, _, _, logits = model(input)
            print(f"Returned logits shape: {logits.shape}")

            logits = torch.squeeze(logits, 1)
            print(f"After squeeze: {logits.shape}")

            logits = torch.mean(logits, 0)
            print(f"After mean: {logits.shape}")

            sig = logits
            print(f"Signature length: {sig.shape[0]}")
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))
            print(f"Current prediction tensor size: {pred.size()}")

        print(f"Final prediction tensor size: {pred.size()}")
        gt = np.load(args.gt, allow_pickle=True)
        print(f"Ground truth size: {len(gt)}")

        pred = list(pred.cpu().detach().numpy())
        print(f"Prediction list size before repeat: {len(pred)}")

        pred = np.repeat(np.array(pred), 16)
        print(f"Prediction array size after repeat: {len(pred)}")

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
    print(f"Test loader created with {len(test_loader)} batches")

    model = model.to(device)
    print("Loading model weights...")
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.testing_model).items()})
    print("Model loaded successfully")

    auc = test(test_loader, model, args, device)
    print(f"Final AUC results: {auc}")

# git config --global user.name "Bardia Soltan"
# git config --global user.email "bardisoltan@gmail.com"
# ssh-add ~/.ssh/id_ed25519