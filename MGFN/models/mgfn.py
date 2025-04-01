import torch
from torch import nn, einsum
from utils.utils import FeedForward, LayerNorm, GLANCE, FOCUS
import option

args = option.parse_args()


def exists(val):
    return val is not None


def attention(q, k, v):
    print(f"attention input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
    sim = einsum('b i d, b j d -> b i j', q, k)
    print(f"sim shape: {sim.shape}")
    attn = sim.softmax(dim=-1)
    print(f"attn shape: {attn.shape}")
    out = einsum('b i j, b j d -> b i d', attn, v)
    print(f"attention output shape: {out.shape}")
    return out


def MSNSD(features, scores, bs, batch_size, drop_out, ncrops, k):
    # magnitude selection and score prediction
    print(
        f"MSNSD input shapes - features: {features.shape}, scores: {scores.shape}, bs: {bs}, batch_size: {batch_size}, ncrops: {ncrops}, k: {k}")
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()
    print(f"Features dimensions: bc={bc}, t={t}, f={f}")

    if ncrops == 1:
        print("Processing with ncrops=1")
        scroes = scores
        normal_features = features[0:batch_size]  # [b/2,32,1024]
        normal_scores = scores[0:batch_size]  # [b/2, 32,1]
        abnormal_features = features[batch_size:]
        abnormal_scores = scores[batch_size:]
        feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b, 32]
        print(f"normal_features shape: {normal_features.shape}, normal_scores shape: {normal_scores.shape}")
        print(f"abnormal_features shape: {abnormal_features.shape}, abnormal_scores shape: {abnormal_scores.shape}")
        print(f"feat_magnitudes shape: {feat_magnitudes.shape}")

    elif ncrops == 10:
        print("Processing with ncrops=10")
        print(f"scores shape before view: {scores.shape}")
        scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
        print(f"scores shape after view and mean: {scores.shape}")
        scores = scores.unsqueeze(dim=2)  # (B,32,1)
        print(f"scores shape after unsqueeze: {scores.shape}")
        normal_features = features[0:batch_size * 10]  # [b/2*ten,32,1024]
        normal_scores = scores[0:batch_size]  # [b/2, 32,1]

        abnormal_features = features[batch_size * 10:]
        abnormal_scores = scores[batch_size:]
        feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
        print(f"feat_magnitudes shape before view: {feat_magnitudes.shape}")
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]
        print(f"feat_magnitudes shape after view and mean: {feat_magnitudes.shape}")
        print(f"normal_features shape: {normal_features.shape}, normal_scores shape: {normal_scores.shape}")
        print(f"abnormal_features shape: {abnormal_features.shape}, abnormal_scores shape: {abnormal_scores.shape}")

    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2
    print(
        f"nfea_magnitudes shape: {nfea_magnitudes.shape}, afea_magnitudes shape: {afea_magnitudes.shape}, n_size: {n_size}")

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        print("Single sample detected (inference mode)")
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes).cuda()
    select_idx = drop_out(select_idx)
    print(f"select_idx shape after dropout: {select_idx.shape}")

    afea_magnitudes_drop = afea_magnitudes * select_idx
    print(f"afea_magnitudes_drop shape: {afea_magnitudes_drop.shape}")
    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    print(f"idx_abn shape: {idx_abn.shape}")
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
    print(f"idx_abn_feat shape after expand: {idx_abn_feat.shape}")

    print(f"abnormal_features shape before view: {abnormal_features.shape}")
    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    print(f"abnormal_features shape after view: {abnormal_features.shape}")
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)
    print(f"abnormal_features shape after permute: {abnormal_features.shape}")

    total_select_abn_feature = torch.zeros(0)
    print(f"Starting total_select_abn_feature shape: {total_select_abn_feature.shape}")
    for i, abnormal_feature in enumerate(abnormal_features):
        print(f"Loop {i}, abnormal_feature shape: {abnormal_feature.shape}")
        feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
        print(f"feat_select_abn shape: {feat_select_abn.shape}")
        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #

    print(f"Final total_select_abn_feature shape: {total_select_abn_feature.shape}")

    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    print(f"idx_abn_score shape: {idx_abn_score.shape}")
    score_abnormal = torch.gather(abnormal_scores, 1, idx_abn_score)
    print(f"score_abnormal after gather: {score_abnormal.shape}")
    score_abnormal = torch.mean(score_abnormal, dim=1)
    print(f"score_abnormal after mean: {score_abnormal.shape}")

    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    select_idx_normal = drop_out(select_idx_normal)
    print(f"select_idx_normal shape: {select_idx_normal.shape}")
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    print(f"idx_normal shape: {idx_normal.shape}")
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])
    print(f"idx_normal_feat shape: {idx_normal_feat.shape}")

    print(f"normal_features shape before view: {normal_features.shape}")
    normal_features = normal_features.view(n_size, ncrops, t, f)
    print(f"normal_features shape after view: {normal_features.shape}")
    normal_features = normal_features.permute(1, 0, 2, 3)
    print(f"normal_features shape after permute: {normal_features.shape}")

    total_select_nor_feature = torch.zeros(0)
    for i, nor_fea in enumerate(normal_features):
        print(f"Loop {i}, nor_fea shape: {nor_fea.shape}")
        feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
        print(f"feat_select_normal shape: {feat_select_normal.shape}")
        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    print(f"Final total_select_nor_feature shape: {total_select_nor_feature.shape}")

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    print(f"idx_normal_score shape: {idx_normal_score.shape}")
    score_normal = torch.gather(normal_scores, 1, idx_normal_score)
    print(f"score_normal after gather: {score_normal.shape}")
    score_normal = torch.mean(score_normal, dim=1)
    print(f"score_normal after mean: {score_normal.shape}")

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    print(
        f"MSNSD output shapes - score_abnormal: {score_abnormal.shape}, score_normal: {score_normal.shape}, abn_feamagnitude: {abn_feamagnitude.shape}, nor_feamagnitude: {nor_feamagnitude.shape}, scores: {scores.shape}")
    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores


class Backbone(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            heads,
            mgfn_type='gb',
            kernel=5,
            dim_headnumber=64,
            ff_repe=4,
            dropout=0.,
            attention_dropout=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads=heads, dim_head=dim_headnumber, local_aggr_kernel=kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads=heads, dim_head=dim_headnumber, dropout=attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding=1),
                attention,
                FeedForward(dim, repe=ff_repe, dropout=dropout),
            ]))

    def forward(self, x):
        print(f"Backbone input shape: {x.shape}")
        for i, (scc, attention, ff) in enumerate(self.layers):
            x_scc = scc(x)
            print(f"Layer {i} - Conv output shape: {x_scc.shape}")
            x = x_scc + x

            x_attn = attention(x)
            print(f"Layer {i} - Attention output shape: {x_attn.shape}")
            x = x_attn + x

            x_ff = ff(x)
            print(f"Layer {i} - FeedForward output shape: {x_ff.shape}")
            x = x_ff + x

        print(f"Backbone output shape: {x.shape}")
        return x


# main class

class mgfn(nn.Module):
    def __init__(
            self,
            *,
            classes=0,
            dims=(64, 128, 1024),
            depths=(args.depths1, args.depths2, args.depths3),
            mgfn_types=(args.mgfn_type1, args.mgfn_type2, args.mgfn_type3),
            lokernel=5,
            channels=2048,  # default
            # channels = 1024,
            ff_repe=4,
            dim_head=64,
            dropout=0.,
            attention_dropout=0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv1d(channels, init_dim, kernel_size=3, stride=1, padding=1)

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim=stage_dim,
                    depth=depth,
                    heads=heads,
                    mgfn_type=mgfn_types,
                    ff_repe=ff_repe,
                    dropout=dropout,
                    attention_dropout=attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride=1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(last_dim)
        )
        self.batch_size = args.batch_size
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(args.dropout_rate)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, video):
        print(f"mgfn input video shape: {video.shape}")
        k = 2
        # (100,10,2048)
        if len(video.size()) == 4:
            bs, ncrops, t, c = video.size()
            print(f"Video dimensions: bs={bs}, ncrops={ncrops}, t={t}, c={c}")
            x = video.view(bs * ncrops, t, c).permute(0, 2, 1)
            print(f"After view and permute: {x.shape}")
        elif len(video.size()) == 3:
            bs, _, _ = video.size()
            ncrops = 1
            x = video.permute(0, 2, 1)
            print(f"After permute (3D input): {x.shape}")

        if x.shape[1] == 2049:
            print("Processing 2049-channel input")
            x_f = x[:, :2048, :]
            print(f"x_f shape: {x_f.shape}")
            x_m = x[:, 2048:, :]
            print(f"x_m shape: {x_m.shape}")
        elif x.shape[1] == 1025:
            print("Processing 1025-channel input")
            x_f = x[:, :1024, :]
            print(f"x_f shape: {x_f.shape}")
            x_m = x[:, 1024:, :]
            print(f"x_m shape: {x_m.shape}")

        x_f = self.to_tokens(x_f)
        print(f"After to_tokens: {x_f.shape}")
        x_m = self.to_mag(x_m)
        print(f"After to_mag: {x_m.shape}")
        x_f = x_f + args.mag_ratio * x_m
        print(f"After adding magnitude: {x_f.shape}")

        for i, (backbone, conv) in enumerate(self.stages):
            print(f"\nStage {i} input shape: {x_f.shape}")
            x_f = backbone(x_f)
            print(f"Stage {i} after backbone: {x_f.shape}")
            if exists(conv):
                x_f = conv(x_f)
                print(f"Stage {i} after conv: {x_f.shape}")

        x_f = x_f.permute(0, 2, 1)
        print(f"After final permute: {x_f.shape}")
        x = self.to_logits(x_f)
        print(f"After to_logits: {x.shape}")
        scores = self.sigmoid(self.fc(x))
        print(f"Scores shape: {scores.shape}")

        score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = MSNSD(x, scores, bs, self.batch_size,
                                                                                         self.drop_out, ncrops, k)
        print(f"Final outputs - score_abnormal: {score_abnormal.shape}, score_normal: {score_normal.shape}")
        print(f"abn_feamagnitude: {abn_feamagnitude.shape}, nor_feamagnitude: {nor_feamagnitude.shape}")
        print(f"scores: {scores.shape}")

        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores
