"""
here is the mian backbone for DMD containing feature decoupling and multimodal transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims # MOSI: 768, 5, 20
        
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.num_of_ha = 8
        self.num_of_hv = 4
        self.linear_expand_size_a = 4
        self.linear_expand_size_v = 2
        self.distribution_dim = 64
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1
        
        # (Optional) Adding dimension adjustment to alleviate the inconsistance of different modalities.
        # Dimension for text modality(MOSI): 16*3*50
        # Dimension for audio modality(MOSI): 16*50*5
        self.tinyhead_a1 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_a2 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_a3 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_a4 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )
        
        self.tinyhead_a5 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_a6 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_a7 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_a8 = nn.Sequential(
            nn.Linear(self.orig_d_a, self.linear_expand_size_a * self.orig_d_a),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_a)
        )

        self.tinyhead_v1 = nn.Sequential(
            nn.Linear(self.orig_d_v, self.linear_expand_size_v * self.orig_d_v),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_v)
        )

        self.tinyhead_v2 = nn.Sequential(
            nn.Linear(self.orig_d_v, self.linear_expand_size_v * self.orig_d_v),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_v)
        )
        
        self.tinyhead_v3 = nn.Sequential(
            nn.Linear(self.orig_d_v, self.linear_expand_size_v * self.orig_d_v),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_v)
        )

        self.tinyhead_v4 = nn.Sequential(
            nn.Linear(self.orig_d_v, self.linear_expand_size_v * self.orig_d_v),
            nn.ReLU(True),
            nn.BatchNorm1d(self.len_v)
        )

        self.head_a, self.head_v = [self.tinyhead_a1, self.tinyhead_a2, self.tinyhead_a3, self.tinyhead_a4, self.tinyhead_a5
                                    , self.tinyhead_a6, self.tinyhead_a7, self.tinyhead_a8], [self.tinyhead_v1, self.tinyhead_v2, self.tinyhead_v3, self.tinyhead_v4]

        # 1. Temporal convolutional layers for initial feature(Multi-head)
        # self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(self.orig_d_a * self.linear_expand_size_a * self.num_of_ha, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v * self.linear_expand_size_v * self.num_of_hv, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)
        
        # 1. Temporal convolutional layers for initial feature(Without multi-head)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # Converting to unified feature space

        # 2.1 Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2.2 Modality-invariant encoder
        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # get distribution from label-based sub block
        self.get_distribution = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), self.distribution_dim)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # 4.2 Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 4. fc layers for homogeneous graph distillation
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        # 5. fc layers for heterogeneous graph distillation
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # 6. Ensemble Projection layers
        # weight for each modality
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        # final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, labels, is_distill=False):
        # print("Original Shape:", "Text:", text.shape, "Video:", video.shape, "Audio:", audio.shape)
        # MOSI: Text-16*3*50, Video-16*50*20, Audio-16*50*5
        # print(self.orig_d_l, self.orig_d_v, self.orig_d_a)
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        # ha, hv = [], []
        # for item in self.head_a:
        #     ha.append(item(audio))
        # for item in self.head_v:
        #     hv.append(item(video))
        # audio, video = torch.cat(ha, dim=-1), torch.cat(hv, dim=-1)
        # print("shape a:", audio.shape, "shape v:", video.shape)
        
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        # Text: 16*768*50, Video: 16*20*50, Audio: 16*5*50
        # print("l_shape:", x_l.shape, "v_shape:", x_v.shape, "a_shape:", x_a.shape)
        
        # Previous MOSEI Conv1d Kernel(l/a/v): 5, 1, 3, config.json Line 86 ~ Line 88.
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        # l/v/a: 16*50*46
        
        # print("x_l:", proj_x_l.shape, "x_v:", proj_x_v.shape, "x_a:", proj_x_a.shape)
        # result1, result2, result3 = proj_x_l[0].cpu().detach().numpy(), proj_x_v[0].cpu().detach().numpy(), proj_x_a[0].cpu().detach().numpy()
        # np.savetxt("x_l.txt", result1)
        # np.savetxt("x_v.txt", result2)
        # np.savetxt("x_a.txt", result3)

        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)
        # print('shape for s_l:', s_l.shape, 's_v:', s_v.shape, 's_a:', s_a.shape) # 16*50*46

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        # print('shape for c_l:', c_l.shape, 'c_v:', c_v.shape, 'c_a:', c_a.shape) #16*50*46
        c_list = [c_l, c_v, c_a]

        # Get postive and negative label index.
        labels_line = labels.view(1, -1).squeeze()
        pos_index = (labels_line >= 0).nonzero().squeeze()
        neg_index = (labels_line < 0).nonzero().squeeze()
        # print(pos_index.size(0))
        if pos_index.numel() > 1:
            if pos_index.size(0) % 2 != 0:
                pos_index = torch.cat((pos_index, pos_index[-1].unsqueeze(0)), dim=0)
        if neg_index.numel() > 1:
            if neg_index.size(0) % 2 != 0:
                neg_index = torch.cat((neg_index, neg_index[-1].unsqueeze(0)), dim=0)
        # print(len(pos_index))

        # (s1.a1 - s1.a2):(s1.v1 - s1.v2):(s1.t1 - s1.t2), a1, a2 means homo and heter features, s1 means same sample.
        s_a_flat, s_v_flat, s_l_flat = s_a.contiguous().view(x_l.size(0), -1), s_v.contiguous().view(x_l.size(0), -1), s_l.contiguous().view(x_l.size(0), -1)
        c_a_flat, c_v_flat, c_l_flat = c_a.contiguous().view(x_l.size(0), -1), c_v.contiguous().view(x_l.size(0), -1), c_l.contiguous().view(x_l.size(0), -1)
        a_sub, v_sub, l_sub = torch.sub(s_a_flat, c_a_flat), torch.sub(s_v_flat, c_v_flat), torch.sub(s_l_flat, c_l_flat)
        # print(a_sub.shape, v_sub.shape, l_sub.shape)
        # a_sub: (s.a1 - s.a2), v_sub: (s.v1 - s.v2), l_sub: (s.t1 - s.t2)
        a_sub_pos, v_sub_pos, l_sub_pos = a_sub[pos_index, :], v_sub[pos_index, :], l_sub[pos_index, :]
        a_sub_neg, v_sub_neg, l_sub_neg = a_sub[neg_index, :], v_sub[neg_index, :], l_sub[neg_index, :]

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        # align: [16*2300] -> [16*50]
        # print('shape for c_l_ain:', c_l_sim.shape, 'c_v_ali:', c_v_sim.shape, 'c_a_ali:', c_a_sim.shape) # 16*50

        # get distribution parameters(mean and var) from network.
        mulogvar_l_pos, mulogvar_v_pos, mulogvar_a_pos = self.get_distribution(l_sub_pos), self.get_distribution(v_sub_pos), self.get_distribution(a_sub_pos)
        mulogvar_l_neg, mulogvar_v_neg, mulogvar_a_neg = self.get_distribution(l_sub_neg), self.get_distribution(v_sub_neg), self.get_distribution(a_sub_neg)

        # concat vector: concat of homo and heter features.
        concat_l = torch.cat([s_l, c_list[0]], dim=1)
        concat_v = torch.cat([s_v, c_list[1]], dim=1)
        concat_a = torch.cat([s_a, c_list[2]], dim=1)
        # print('shape for concat:', concat_l.shape)
        recon_l = self.decoder_l(concat_l)
        recon_v = self.decoder_v(concat_v)
        recon_a = self.decoder_a(concat_a)
        # print('shape for recon_l:', recon_l.shape, 'recon_v:', recon_v.shape, 'recon_a:', recon_a.shape) # 16*50*46

        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)
        # print('shape for s_l_r:', s_l_r.shape, 's_v_r:', s_v_r.shape, 's_a_r:', s_a_r.shape) # 16*50*46

        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        proj_s_l = self.proj_cosine_l(s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_v = self.proj_cosine_v(s_v.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_a = self.proj_cosine_a(s_a.transpose(0, 1).contiguous().view(x_l.size(0), -1))

        c_l_att = self.self_attentions_c_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]
        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        # cross-modal attention
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        last_h_l = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion = torch.sigmoid(self.weight_c(c_fusion))

        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'logits_l_homo': logits_l_low,
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            'repr_l_homo': repr_l_low,
            'repr_v_homo': repr_v_low,
            'repr_a_homo': repr_a_low,
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'proj_s_l': proj_s_l,
            'proj_s_v': proj_s_v,
            'proj_s_a': proj_s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            'mulogvar_l_pos': mulogvar_l_pos,
            'mulogvar_v_pos': mulogvar_v_pos,
            'mulogvar_a_pos': mulogvar_a_pos,
            'mulogvar_l_neg': mulogvar_l_neg,
            'mulogvar_v_neg': mulogvar_v_neg,
            'mulogvar_a_neg': mulogvar_a_neg,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'concat_l': concat_l,
            'concat_v': concat_v,
            'concat_a': concat_a,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'c_l_att': c_l_att,
            'c_v_att': c_v_att,
            'c_a_att': c_a_att,
            'logits_l_hetero': logits_l_high,
            'logits_v_hetero': logits_v_high,
            'logits_a_hetero': logits_a_high,
            'repr_l_hetero': hs_proj_l_high,
            'repr_v_hetero': hs_proj_v_high,
            'repr_a_hetero': hs_proj_a_high,
            'last_h_l': h_ls[-1],
            'last_h_v': h_vs[-1],
            'last_h_a': h_as[-1],
            'logits_c': logits_c,
            'output_logit': output
        }
        return res
