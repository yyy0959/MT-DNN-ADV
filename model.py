import torch.nn as nn
from transformers import BertModel
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch

class diff_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, S, T):
        res = torch.matmul(S, T.T)
        dif_loss = torch.norm(res, p='fro', dim=None, keepdim=False, out=None, dtype=None)
        return dif_loss / (S.size(0) * T.size(0))

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output

class Bert(nn.Module):
    def __init__(self, dataset_names=None, model_name='bert-base-uncased', need_transfer_test=False):
        super().__init__()
        self.datasets_names = dataset_names
        self.num_task = len(dataset_names)
        self.taskdic = {}
        self.need_transfer_test = need_transfer_test
        for i in range(len(dataset_names)):
            self.taskdic[dataset_names[i]] = i
        if dataset_names==None or "mrpc" in dataset_names:
            self.mrpc_bert = BertModel.from_pretrained(model_name)
            self.linear_for_mrpc = nn.Linear(768 * 2, 2)
        if dataset_names == None or "stsb" in dataset_names:
            self.stsb_bert = BertModel.from_pretrained(model_name)
            self.linear_for_stsb = nn.Linear(768 * 2, 6)
        if dataset_names == None or "sst2" in dataset_names:
            self.sst2_bert = BertModel.from_pretrained(model_name)
            self.linear_for_sst2 = nn.Linear(768 * 2, 2)
        if dataset_names == None or "cola" in dataset_names:
            self.cola_bert = BertModel.from_pretrained(model_name)
            self.linear_for_cola = nn.Linear(768 * 2, 2)
        if dataset_names == None or "qnli" in dataset_names:
            self.qnli_bert = BertModel.from_pretrained(model_name)
            self.linear_for_qnli = nn.Linear(768 * 2, 2)
        if dataset_names == None or "qqp" in dataset_names:
            self.qqp_bert = BertModel.from_pretrained(model_name)
            self.linear_for_qqp = nn.Linear(768 * 2, 2)
        if dataset_names == None or "mnli" in dataset_names:
            self.mnli_bert = BertModel.from_pretrained(model_name)
            self.linear_for_mnli = nn.Linear(768 * 2, 3)
        if dataset_names == None or "rte" in dataset_names:
            self.rte_bert = BertModel.from_pretrained(model_name)
            self.linear_for_rte = nn.Linear(768 * 2, 2)
        if dataset_names == None or "wnli" in dataset_names:
            self.wnli_bert = BertModel.from_pretrained(model_name)
            self.linear_for_wnli = nn.Linear(768 * 2, 2)
        if dataset_names == None or "ax" in dataset_names:
            self.ax_bert = BertModel.from_pretrained(model_name)
            self.linear_for_ax = nn.Linear(768 * 2, 3)
        if need_transfer_test:
            self.linear_for_srte = nn.Linear(768, 2)
        # if dataset_names == None or "onto" in dataset_names:
        #     self.onto_bert = BertModel.from_pretrained(model_name)
        #     self.lstm_for_onto = nn.LSTM(768 * 2, 36, batch_first=True, bidirectional=True)
        #     self.linear_for_onto = nn.Linear(2 * 36, 36)
        #     self.crf_for_onto = CRF(36, batch_first=True)
        # if dataset_names == None or "wmt19" in dataset_names:
        #     self.wmt_bert = BertModel.from_pretrained(model_name)
        #     self.decoder = TransformerDecoder(d_model=768 * 2, nhead=4, num_layers=8, dim_feedforward=256, dropout=0.1)
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, self.num_task)


    def forward(self, input_ids, input_mask, type_n, ner_labels=None, decoder_input_ids=None, device=None):
        if type_n != "srte":
            type_id = self.taskdic[type_n]
        shared_feature = self.bert(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        clf_feature_shared = shared_feature[:, 0, :]
        clf_feature_shared = clf_feature_shared.view(clf_feature_shared.shape[0], -1)
        if type_n == "mrpc":
            out_mrpc = self.mrpc_bert(input_ids=input_ids, attention_mask=input_mask)
            out_mrpc = out_mrpc.last_hidden_state[:, 0, :]
            out_mrpc = out_mrpc.view(out_mrpc.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_mrpc), dim=1)
            out = self.linear_for_mrpc(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_mrpc
        elif type_n == "stsb":
            out_stsb = self.stsb_bert(input_ids=input_ids, attention_mask=input_mask)
            out_stsb = out_stsb.last_hidden_state[:, 0, :]
            out_stsb = out_stsb.view(out_stsb.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_stsb), dim=1)
            out = self.linear_for_stsb(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_stsb
        elif type_n == "sst2":
            out_sst2 = self.sst2_bert(input_ids=input_ids, attention_mask=input_mask)
            out_sst2 = out_sst2.last_hidden_state[:, 0, :]
            out_sst2 = out_sst2.view(out_sst2.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_sst2), dim=1)
            out = self.linear_for_sst2(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_sst2
        elif type_n == "cola":
            out_cola = self.cola_bert(input_ids=input_ids, attention_mask=input_mask)
            out_cola = out_cola.last_hidden_state[:, 0, :]
            out_cola = out_cola.view(out_cola.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_cola), dim=1)
            out = self.linear_for_cola(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_cola
        elif type_n == "qnli":
            out_qnli = self.qnli_bert(input_ids=input_ids, attention_mask=input_mask)
            out_qnli = out_qnli.last_hidden_state[:, 0, :]
            out_qnli = out_qnli.view(out_qnli.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_qnli), dim=1)
            out = self.linear_for_qnli(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_qnli
        elif type_n == "qqp":
            out_qqp = self.qqp_bert(input_ids=input_ids, attention_mask=input_mask)
            out_qqp = out_qqp.last_hidden_state[:, 0, :]
            out_qqp = out_qqp.view(out_qqp.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_qqp), dim=1)
            out = self.linear_for_qqp(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_qqp
        elif type_n == "mnli":
            out_mnli = self.mnli_bert(input_ids=input_ids, attention_mask=input_mask)
            out_mnli = out_mnli.last_hidden_state[:, 0, :]
            out_mnli = out_mnli.view(out_mnli.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_mnli), dim=1)
            out = self.linear_for_mnli(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_mnli
        elif type_n == "rte":
            out_rte = self.rte_bert(input_ids=input_ids, attention_mask=input_mask)
            out_rte = out_rte.last_hidden_state[:, 0, :]
            out_rte = out_rte.view(out_rte.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_rte), dim=1)
            out = self.linear_for_rte(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_rte
        elif type_n == "wnli":
            out_wnli = self.wnli_bert(input_ids=input_ids, attention_mask=input_mask)
            out_wnli = out_wnli.last_hidden_state[:, 0, :]
            out_wnli = out_wnli.view(out_wnli.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_wnli), dim=1)
            out = self.linear_for_wnli(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_wnli
        elif type_n == "ax":
            out_ax = self.ax_bert(input_ids=input_ids, attention_mask=input_mask)
            out_ax = out_ax.last_hidden_state[:, 0, :]
            out_ax = out_ax.view(out_ax.shape[0], -1)
            out = torch.cat((clf_feature_shared, out_ax), dim=1)
            out = self.linear_for_ax(out)
            out_shared = GradientReverseFunction.apply(clf_feature_shared)
            cls_type = self.classifier(out_shared)
            return out, cls_type, type_id, clf_feature_shared, out_ax
        elif type_n == "srte":
            out = self.linear_for_srte(clf_feature_shared)
            return out
        # elif type_n == "onto":
        #     type_id = 2
        #     out_onto = self.onto_bert(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        #     out = torch.cat((shared_feature, out_onto), dim=2)
        #     out, state = self.lstm_for_onto(out)
        #     out = self.linear_for_onto(out)
        #     out_shared = GradientReverseFunction.apply(shared_feature)
        #     cls_type = self.classifier(out_shared)
        #     cls_type = cls_type.view(cls_type.shape[0]*cls_type.shape[1], cls_type.shape[2])
        #     if ner_labels != None:
        #         logits = self.crf(out, ner_labels)
        #         return -1.0 * logits, self.crf.decode(out), cls_type, type_id, shared_feature, out_onto
        #     else:
        #         return self.crf.decode(out)
        # elif type_n == "wmt19":
        #     type_id = 3
        #     out_wmt = self.wmt_bert(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        #     out = torch.cat((shared_feature, out_wmt), dim=2)
        #     decoder_outputs = self.decoder(decoder_input_ids, out,
        #                                     tgt_mask=self.decoder.generate_square_subsequent_mask(decoder_input_ids.size(1)).to(device),
        #                                     memory_mask=None,
        #                                     tgt_key_padding_mask=None,
        #                                     memory_key_padding_mask=None)
        #     logits = decoder_outputs.transpose(0, 1)
        #     out_shared = GradientReverseFunction.apply(shared_feature)
        #     cls_type = self.classifier(out_shared)
        #     cls_type = cls_type.view(cls_type.shape[0] * cls_type.shape[1], cls_type.shape[2])
        #     return logits, cls_type, type_id, clf_feature_shared, out_wmt
        return None