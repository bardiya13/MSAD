import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MGFN')
    parser.add_argument('--feat_extractor', default='i3d', choices=['i3d', 'c3d', 'swin'])
    parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: UCF:2048//xd:1024)')
    parser.add_argument('--hiddensize', type=int, default=512, help='size of feature (default: 512)')
    parser.add_argument('--modality', default='RGB', help='the type of the input, RGB, AUDIO, or MIX')
    parser.add_argument('--rgb-list', default='/kaggle/working/train_folder_list.txt', help='list of rgb features ')
    parser.add_argument('--test-rgb-list', default='/kaggle/working/test_folder_list.txt', help='list of test rgb features')
    parser.add_argument('--gt', default="/kaggle/working/ground_truth_train_new.npy", help='file of ground truth ')
    parser.add_argument('--mag_ratio', type=float, default=0.1, help='mag ratio')
    parser.add_argument('--comment', default='mgfn', help='comment for the ckpt name of the training')

    parser.add_argument('--test_feature_address', type=str, default="/kaggle/input/kkkkkkkkk/feature_UCSD2/output", help='where did you store your test features?')
    parser.add_argument('--train_feature_address', type=str, default="/kaggle/input/kkkkkkkkk/train_feature_UCSD2/output_train", help='where did you store your train features?')
    parser.add_argument('--test_label_address', type=str, default="/kaggle/input/ucsd-p2/test_labels_new/test_labels_new", help='where did you store your test labels?')
    parser.add_argument('--train_label_address', type=str, default="/kaggle/input/ucsd-p2/train_labels_new/train_labels_new", help='where did you store your train labels?')


    parser.add_argument('--seg_length', type=int, default=32, help='default:32')
    parser.add_argument('--local_con', default='static', help='dynamic/static')
    #for dynamic
    parser.add_argument('--head_K', type=int, default=4, help='default = 4')
    #model structure
    parser.add_argument('--depths1', type=int, default=3, help='depths1')
    parser.add_argument('--depths2', type=int, default=3, help='depths2')
    parser.add_argument('--depths3', type=int, default=2, help='depths3')

    parser.add_argument('--mgfn_type1', default='gb', help='mgfn_types1')
    parser.add_argument('--mgfn_type2', default='fb', help='mgfn_types2')
    parser.add_argument('--mgfn_type3', default='fb', help='mgfn_types3')

    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--gpus', type=str, default='0', help='gpus')
    parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form) default:0.001')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')
    parser.add_argument('--workers', default=0, help='number of workers in dataloader')
    parser.add_argument('--model-name', default='mgfn', help='name to save model')
    parser.add_argument('--pretrained_ckpt', default= None, help='ckpt for pretrained model')
    parser.add_argument('--num-classes', type=int, default=2, help='number of class')
    parser.add_argument('--datasetname', default='MSAD', help='dataset to train on (default:UCF/XD/UCF-bg-fg-sepa )')  # !!!
    parser.add_argument('--preprocessed', action = 'store_true', help='if train set is already segmented')
    parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
    parser.add_argument('--max-epoch', type=int, default=700, help='maximum iteration to train (default: 100)')
    parser.add_argument('--testing-model', default=None, help='The model used for testing.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.gpus = [i for i in range(len(args.gpus.split(',')))]

    return args