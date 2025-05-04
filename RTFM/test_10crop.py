import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from tqdm import tqdm

def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh2.npy')
        if args.dataset == 'ped2':
            gt = np.load('list/gt-ped2.npy')
        if args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        if args.dataset == 'msad':
            gt = np.load('/kaggle/working/MSAD/RTFM/list/gt-MSAD-WS-new.npy')
        if args.dataset == 'cuhk':
            gt = np.load('list/gt-cuhk.npy')
        kk = 0
        gt_new = []
        for i, inputs in tqdm(enumerate(dataloader)):
            print(i)

            input = inputs.to(device)
            if sum(gt[kk:kk+inputs.shape[1]]) == 0:
                continue

            gt_new.append(gt[kk:kk+inputs.shape[1]])
            kk += input.shape[1]


        if len(input.size()) == 4:
            input = input.permute(0, 2, 1, 3)
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input)
        logits = torch.squeeze(logits, 1)
        logits = torch.mean(logits, 0)
        sig = logits
            # featurelen.append(len(sig))
        pred = torch.cat((pred, sig))

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))


        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        print("gt_new_len",len(gt_new))
        # gt_new=np.array(gt)

        fpr, tpr, threshold = roc_curve(list(gt_new), pred)

        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt_new), pred)
        pr_auc = auc(recall, precision)
        # print('pr_auc : ' + str(rec_auc))
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        return rec_auc
#
# import matplotlib.pyplot as plt
# import torch
# from sklearn.metrics import auc, roc_curve, precision_recall_curve
# import numpy as np
# from tqdm import tqdm
#
#
# def test(dataloader, model, args, device):
#     with torch.no_grad():
#         model.eval()
#         pred = torch.zeros(0, device=device)
#
#         for i, inputs in tqdm(enumerate(dataloader)):
#
#             # (1, T, 1024)
#             input = inputs.to(device)
#             # print(inputs[0].shape)
#             # (B, 10, T, 2048) -> (B, T, 10, 2048)
#             if len(input.size()) == 4:
#                 input = input.permute(0, 2, 1, 3)
#             score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(
#                 input)
#             logits = torch.squeeze(logits, 1)
#             logits = torch.mean(logits, 0)
#             sig = logits
#             # featurelen.append(len(sig))
#             pred = torch.cat((pred, sig))
#
#             if args.dataset == 'shanghai':
#                 gt = np.load('list/gt-sh2.npy')
#             if args.dataset == 'ped2':
#                 gt = np.load('list/gt-ped2.npy')
#             if args.dataset == 'ucf':
#                 gt = np.load('list/gt-ucf.npy')
#             if args.dataset == 'msad':
#                 gt = np.load('list/gt-MSAD-WS-new.npy')
#             if args.dataset == 'cuhk':
#                 gt = np.load('list/gt-cuhk.npy')
#
#             pred = list(pred.cpu().detach().numpy())
#             pred = np.repeat(np.array(pred), 16)
#
#
#             fpr, tpr, threshold = roc_curve(list(gt), pred)
#             rec_auc = auc(fpr, tpr)
#             print('auc : ' + str(rec_auc))
#
#             precision, recall, th = precision_recall_curve(list(gt), pred)
#             pr_auc = auc(recall, precision)
#             # print('pr_auc : ' + str(rec_auc))
#             # viz.plot_lines('pr_auc', pr_auc)
#             # viz.plot_lines('auc', rec_auc)
#             # viz.lines('scores', pred)
#             # viz.lines('roc', tpr, fpr)
#             return rec_auc
#
#
