from cProfile import label
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from .distribution import gussian_kld
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
from .model.InfoNCE import InfoNCE
from .model.InfoNCE import InfoNCE_mul
from torch.utils.tensorboard import SummaryWriter
import hypertools as hyp
import matplotlib.pyplot as plt

logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse
    
def flatten(x, len):
    return x.transpose(0,1).contiguous().view(len,-1)

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.writer = SummaryWriter("./logs")

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model[0].parameters()) + \
                 list(model[1].parameters()) + \
                 list(model[2].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net
        
        best_test_acc2, best_test_acc7, best_test_f1 = 0.0, 0.0, 0.0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            self.data, self.vis_labels = [], []
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    # Text: 16*3*50 Video: 16*50*20 Audio: 16*50*5
                    # print("Original Shape:", "Text:", text.shape, "Video:", vision.shape, "Audio:", audio.shape)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    output = model[0](text, audio, vision, labels, is_distill=True)

                    # logits for homo GD
                    logits_homo.append(output['logits_l_homo'])
                    logits_homo.append(output['logits_v_homo'])
                    logits_homo.append(output['logits_a_homo'])

                    # reprs for homo GD
                    reprs_homo.append(output['repr_l_homo'])
                    reprs_homo.append(output['repr_v_homo'])
                    reprs_homo.append(output['repr_a_homo'])

                    # logits for hetero GD
                    logits_hetero.append(output['logits_l_hetero'])
                    logits_hetero.append(output['logits_v_hetero'])
                    logits_hetero.append(output['logits_a_hetero'])

                    # reprs for hetero GD
                    reprs_hetero.append(output['repr_l_hetero'])
                    reprs_hetero.append(output['repr_v_hetero'])
                    reprs_hetero.append(output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)

                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)

                    # distribution parameters(mu and logvar)
                    mulogvar_l_pos = output['mulogvar_l_pos']
                    mulogvar_v_pos = output['mulogvar_v_pos']
                    mulogvar_a_pos = output['mulogvar_a_pos']
                    mulogvar_l_neg = output['mulogvar_l_neg']
                    mulogvar_v_neg = output['mulogvar_v_neg']
                    mulogvar_a_neg = output['mulogvar_a_neg']
                    # print(mulogvar_a_neg.shape, mulogvar_a_pos.shape)
                    dis_list = [mulogvar_l_pos, mulogvar_v_pos, mulogvar_a_pos, mulogvar_l_neg, mulogvar_v_neg, mulogvar_a_neg]
                    loss_kl = 0.0
                    num = 0
                    for item in dis_list:
                        # print('elements:', len(item.shape))
                        if item.numel() > 1:
                            if item.shape[0] >= 4 and len(item.shape) > 1:
                                # print(item.shape[0], item.shape[1])
                                mu, var = torch.split(item, int(item.shape[1]/2), dim=1) # mu: num_of_pos(neg)*32, var: num_of_pos(neg)*32
                                # print(mu.shape, var.shape)
                                mu1, mu2 = torch.split(mu, int(mu.shape[1]/2), dim=1)
                                # print(mu1.shape, mu2.shape)
                                var1, var2 = torch.split(var, int(var.shape[1]/2), dim=1)
                                # print(var1.shape, var2.shape)
                                kld = gussian_kld(mu1, mu2, var1.abs(), var2.abs())
                                loss_kl += kld
                                num += 1
                    if num > 0:
                        loss_kl = 0.005 * (loss_kl / num)
                    if loss_kl >= 100:
                        loss_kl = 0.0
                    # print(loss_kl)

                    # edges for homo distill
                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # reconstruction loss
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    output['s_l'], output['s_v'], output['s_a'] = flatten(output['s_l'], labels.size(0)), flatten(output['s_v'], labels.size(0)), flatten(output['s_a'], labels.size(0))
                    output['c_l'], output['c_v'], output['c_a'] = flatten(output['c_l'], labels.size(0)), flatten(output['c_v'], labels.size(0)), flatten(output['c_a'], labels.size(0))
                    # print(output['s_l'].shape)
                    cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_a = self.cosine(output['s_a'], output['c_a'],
                                                          torch.tensor([-1]).cuda()).mean(0)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # Adding recrusive loss.
                    # There are two pairs of contrastive learning:
                    # Group 1: anchor-X(com), neg-Dec(X(com)), pos-Enc(Dec(X(com))), which is replacing Loss(cyc)
                    # Gropu 2: anchor-X(proj), neg-Enc(X(proj)), pos-Dec(Enc(x(proj))), which is replacing Loss(rec)
                    loss_NCE = InfoNCE()
                    # Gropu 1 implementation:
                    query_l, query_v, query_a = flatten(output['s_l'], labels.size(0)), flatten(output['s_v'], labels.size(0)), flatten(output['s_a'], labels.size(0))
                    pos_l, pos_v, pos_a = flatten(output['s_l_r'], labels.size(0)), flatten(output['s_v_r'], labels.size(0)), flatten(output['s_a_r'], labels.size(0))
                    neg_l, neg_v, neg_a = flatten(output['recon_l'], labels.size(0)), flatten(output['recon_v'], labels.size(0)), flatten(output['recon_a'], labels.size(0))
                    loss_cyc = 0.0
                    # print(query_l.shape, query_v.shape, query_a.shape)
                    loss_cyc_l = loss_NCE(query_l, pos_l, neg_l)
                    loss_cyc_v = loss_NCE(query_v, pos_v, neg_v)
                    loss_cyc_a = loss_NCE(query_a, pos_a, neg_a)
                    loss_cyc = loss_cyc + loss_cyc_l + loss_cyc_v + loss_cyc_a
                    # Group 2 implementation:
                    query_l, query_v, query_a = flatten(output['origin_l'], labels.size(0)), flatten(output['origin_v'], labels.size(0)), flatten(output['origin_a'], labels.size(0))
                    neg_l_homo, neg_v_homo, neg_a_homo = flatten(output['c_l'], labels.size(0)), flatten(output['c_v'], labels.size(0)), flatten(output['c_a'], labels.size(0))
                    neg_l_heter, neg_v_heter, neg_a_heter = flatten(output['s_l'], labels.size(0)), flatten(output['s_v'], labels.size(0)), flatten(output['s_a'], labels.size(0))
                    pos_l, pos_v, pos_a = flatten(output['recon_l'], labels.size(0)), flatten(output['recon_v'], labels.size(0)), flatten(output['recon_a'], labels.size(0))
                    loss_rec = 0.0
                    # Adding parameters before modalities is available.
                    loss_homo = loss_NCE(query_l, pos_l, neg_l_homo) + loss_NCE(query_v, pos_v, neg_v_homo) + loss_NCE(query_a, pos_a, neg_a_homo)
                    loss_heter = loss_NCE(query_l, pos_l, neg_l_heter) + loss_NCE(query_v, pos_v, neg_v_heter) + loss_NCE(query_a, pos_a, neg_a_heter)
                    loss_rec = loss_rec + 0.2 * loss_homo + 0.8 * loss_heter
                    loss_recursive = loss_rec
                    # print("loss_recr:", loss_recursive)
                    
                    # Visualization part, can be transfered to other parts.
                    for item in output['s_l']:
                        self.data.append(item.cpu().detach().numpy())
                    labels_list = labels.view(1, -1).squeeze(-1).cpu().detach().numpy()
                    labels_list = labels_list[0]
                    for item in labels_list:
                        if item < 0 and item >= -1:
                            self.vis_labels.append('weak negative')
                        elif item < -1:
                            self.vis_labels.append('strong negative')
                        elif item >= 0 and item < 1:
                            self.vis_labels.append('weak positive')
                        else:
                            self.vis_labels.append('strong positive')
                    # print(self.vis_labels)
                    # print(type(self.vis_labels))
                    # End of visualization part.
                    # vis_labels.append(item.astype(float) for item in labels.view(1, -1).cpu().detach().numpy())
                    # print(data)
                    # print(labels_list)
                    # print("NCE:", output['s_l'].shape, output['s_v'].shape, output['s_a'].shape)

                    # Replace ort loss by InfoNCE loss
                    # Positive sample: the sample in same representation space(homo/heter) with different modalities
                    # Negative samples: the sample in different representation space with same modality
                    output['s_l'] = output['s_l'].transpose(0,1).contiguous().view(labels.size(0),-1)
                    output['s_v'] = output['s_v'].transpose(0,1).contiguous().view(labels.size(0),-1)
                    output['s_a'] = output['s_a'].transpose(0,1).contiguous().view(labels.size(0),-1)
                    output['c_l'] = output['c_l'].transpose(0,1).contiguous().view(labels.size(0),-1)
                    output['c_v'] = output['c_v'].transpose(0,1).contiguous().view(labels.size(0),-1)
                    output['c_a'] = output['c_a'].transpose(0,1).contiguous().view(labels.size(0),-1)
                    query_c, query_s = output['c_v'], output['s_v']
                    pos_c, pos_s = output['c_a'], output['s_a']
                    loss_nce = 0.0
                    neg_c, neg_s = output['s_v'], output['c_v']
                    loss_NCE = InfoNCE()
                    loss_nce_c = loss_NCE(query_c, pos_c, neg_c)
                    loss_nce_s = loss_NCE(query_s, pos_s, neg_s)
                    loss_nce = loss_nce + loss_nce_c + loss_nce_s
                    #     neg_c.append()
                    # # print("labels:", labels)
                    # for i in range(labels.size(0)):
                    #     loss_NCE = InfoNCE()
                    #     neg_c, neg_s = output['s_l'], output['c_l']
                    #     loss_nce_c = loss_NCE(query_c, pos_c, neg_c)
                    #     loss_nce_s = loss_NCE(query_s, pos_s, neg_s)
                    #     loss_nce = loss_nce + loss_nce_c + loss_nce_s
                    # print("loss_nce:", loss_nce)
                    
                    # Replace GD(homo/hetre) with Contrastive loss
                    # print("output shape:", output['s_l'].shape, output['c_l_att'].shape, output['c_v_att'].shape, output['c_a_att'].shape,
                    #       output['last_h_l'].shape, output['last_h_v'].shape, output['last_h_a'].shape,
                    #       "label shape:", labels.shape)
                    output['c_l_att'] = output['c_l_att'].transpose(0,1).contiguous().view(labels.size(0), -1)
                    output['c_v_att'] = output['c_v_att'].transpose(0,1).contiguous().view(labels.size(0), -1)
                    output['c_a_att'] = output['c_a_att'].transpose(0,1).contiguous().view(labels.size(0), -1)
                    output['last_h_l'] = output['last_h_l'].transpose(0,1).contiguous().view(labels.size(0), -1)
                    output['last_h_v'] = output['last_h_v'].transpose(0,1).contiguous().view(labels.size(0), -1)
                    output['last_h_a'] = output['last_h_a'].transpose(0,1).contiguous().view(labels.size(0), -1)
                    batch_data_c_l, batch_data_c_v, batch_data_c_a = output['c_l_att'], output['c_v_att'], output['c_a_att']
                    batch_data_h_l, batch_data_h_v, batch_data_h_a = output['last_h_l'], output['last_h_v'], output['last_h_a']
                    loss_c, loss_h = 0.0, 0.0
                    loss_nce_mul = InfoNCE_mul()
                    # Same modality, push away different emotion features, pull close similar emotion features.
                    loss_nce_c_l, loss_nce_c_v, loss_nce_c_a = loss_nce_mul(batch_data_c_l, labels), loss_nce_mul(batch_data_c_v, labels), loss_nce_mul(batch_data_c_a, labels)
                    loss_nce_h_l, loss_nce_h_v, loss_nce_h_a = loss_nce_mul(batch_data_h_l, labels), loss_nce_mul(batch_data_h_v, labels), loss_nce_mul(batch_data_h_a, labels)
                    loss_c = loss_c + loss_nce_c_l + loss_nce_c_v + loss_nce_c_a
                    loss_h = loss_h + loss_nce_h_l + loss_nce_h_v + loss_nce_h_a
                    # print("loss_c:", loss_c, "loss_h:", loss_h)

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # homo GD loss
                    # loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                    #     model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    # graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # # # hetero GD loss
                    # loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                    #     model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    # graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    self.writer.add_scalars('Loss Analysis', 
                                            {
                                                'loss_task': loss_task,
                                                'loss_rec': loss_rec,
                                                'loss_cyc': loss_cyc,
                                                'loss_recursive': 0.1 * loss_recursive,
                                                'loss_sim': loss_sim,
                                                'loss_c':loss_c * 0.05,
                                                'loss_h':loss_h * 0.05
                                            }, epochs)
                    # combined_loss = loss_task + \
                    #                 graph_distill_loss_homo + graph_distill_loss_hetero
                                    # (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1
                    
                    combined_loss = loss_task + loss_kl * 0.05
                        # loss_task + loss_recursive * 0.05 + loss_sim * 0.01
                        # loss_nce * 0.1
                        # 0.05 * loss_recursive

                    # logger.info(
                    #     f"-- loss_task: {loss_task} "
                    #     f"-- loss_nce: {loss_nce}"
                    # )
                    #     f"-- Distill_loss_homo: {graph_distill_loss_homo} "
                    #     f"-- Distill_loss_hete: {graph_distill_loss_hetero} "
                    #     f"-- loss_s_sr: {loss_s_sr} "
                    #     f"-- loss_recon: {loss_recon}"
                    #     f"-- loss_sim: {loss_sim}"
                    #     f"-- loss_ort: {loss_ort}"
                    #     f"-- loss_nce: {loss_nce}"
                    # )+26
                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters()) + \
                                 list(model[1].parameters()) + \
                                 list(model[2].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
                self.data = np.array(self.data)
                self.vis_labels = np.array(self.vis_labels)
                # print("type:", type(self.data), type(self.vis_labels))
                # self.visualization(epoch=epochs)

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f">> rec_loss: {loss_rec} "
                f">> cyc_loss: {loss_cyc}"
                f">> kl_loss: {loss_kl}"
                # f">> MLTC_loss_c: {loss_c}"
                # f">> MLTC_loss_h: {loss_h}"
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            if test_results['Acc_2'] > best_test_acc2:
                best_test_acc2 = test_results['Acc_2']
            if test_results['Acc_7'] > best_test_acc7:
                best_test_acc7 = test_results['Acc_7']
            if test_results['F1_score'] > best_test_f1:
                best_test_f1 = test_results['F1_score']
            print("best_test_results_Acc2:", best_test_acc2, "best_test_Acc7:", best_test_acc7, "best_test_F1:", best_test_f1)
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                print("better results")
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/dmd_15_best.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
                with open("./results_original.txt", 'a+', encoding='utf-8') as fp:
                    fp.write("Loss:" + train_loss + "Train:" + train_results + "Vaild:"
                             + val_results + "Test:" + test_results + '\n')
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
            
            # if epochs > self.args.total_epoches:
            #     return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision, labels, is_distill=True)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
    
    # Visualization for different representation distribution.
    # data: original data, labels: sentiment label corresponding to original data
    # We must pass labels the same number of rows as in our data matrix.
    def visualization(self, save_dir='./visualization', epoch=0):
        raw_data, raw_labels = self.data, self.vis_labels
        # print("actual type:", type(raw_data), type(raw_labels), raw_labels)
        hyp.plot(raw_data, '.', reduce='TSNE', ndims=2, hue=raw_labels, legend=True)
        plt.title('TSNE')
        savepath = save_dir + os.sep + str(epoch)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.savefig(str(savepath + os.sep + str(epoch) + 'baseline_GD' + '.jpg'))
