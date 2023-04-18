import torch.optim as optim
from dataset import *
from model import *
import argparse
from math import sqrt
from scipy.stats import pearsonr


def MCC(TP, FP, FN, TN):
    numerator = (TP * TN) - (FP * FN)
    denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    result = numerator/(denominator+1e-5)
    return result


def pearson(desc1, desc2):
    x_ = desc1 - np.mean(desc1)
    y_ = desc2 - np.mean(desc2)
    r = np.dot(x_,y_) / (np.linalg.norm(x_)*np.linalg.norm(y_))
    return r


def parse_args():
    parser = argparse.ArgumentParser(description="argparse")
    parser.add_argument('--datasets',default="mrpc,stsb,sst2,qnli,wnli,rte,cola", help = "the datasets to be chosen for training")
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_epoch', default=15, type=int)
    parser.add_argument('--need_transfer', default=1, type=int)
    parser.add_argument('--use_adv', default=1, type=int)
    parser.add_argument('--use_diff', default=0, type=int)
    args = parser.parse_args()
    return args

def val_and_test_mcc(val_dataloader, test_dataloader, device, model, name):
    TP, FP, FN, TN = 0,0,0,0
    for idx, mask, target in val_dataloader:
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output, cls_result, type_id, shared, specific = model(idx, mask, name)
        pred = output.argmax(dim=1)
        for i in range(pred.size(0)):
            if pred[i] == 1 and target[i] == 1:
                TP += 1
            elif pred[i] == 1 and target[i] == 0:
                FP += 1
            elif pred[i] == 0 and target[i] == 1:
                FN += 1
            elif pred[i] == 0 and target[i] == 0:
                TN += 1
    val_mcc = MCC(TP, FP, FN, TN)
    print(name + ' Val MCC: {:.6f}'.format(val_mcc))

    # TP, FP, FN, TN = 0, 0, 0, 0
    # for idx, mask, target in test_dataloader:
    #     idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
    #     output, cls_result, type_id, shared, specific = model(idx, mask, name)
    #     pred = output.argmax(dim=1)
    #     for i in range(pred.size(0)):
    #         if pred[i] == 1 and target[i] == 1:
    #             TP += 1
    #         elif pred[i] == 1 and target[i] == 0:
    #             FP += 1
    #         elif pred[i] == 0 and target[i] == 1:
    #             FN += 1
    #         elif pred[i] == 0 and target[i] == 0:
    #             TN += 1
    # test_mcc = MCC(TP, FP, FN, TN)
    # print(name + ' Test MCC: {:.6f}'.format(test_mcc))
    return [val_mcc]


def val_and_test(val_dataloader, test_dataloader, device, model, name):
    correct = 0
    total_num = 0
    for idx, mask, target in val_dataloader:
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output, cls_result, type_id, shared, specific = model(idx, mask, name)
        pred = output.argmax(dim=1)
        correct += int(pred.eq(target).sum())
        total_num += pred.size(0)
    val_acc = correct / total_num
    print(name + ' Val Accuracy: {:.6f}'.format(correct / total_num))

    correct = 0
    total_num = 0
    for idx, mask, target in test_dataloader:
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output, cls_result, type_id, shared, specific = model(idx, mask, name)
        pred = output.argmax(dim=1)
        correct += int(pred.eq(target).sum())
        total_num += pred.size(0)
    test_acc = correct / len(test_dataloader)
    print(name + ' Test Accuracy: {:.6f}'.format(correct / total_num))
    return [val_acc, test_acc]


def val_and_test_pearson(val_dataloader, test_dataloader, device, model, name, num_class):
    total_num = 0
    total_pearson = 0
    for idx, mask, target in val_dataloader:
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output, cls_result, type_id, shared, specific = model(idx, mask, name)
        output = torch.softmax(output, dim=-1)
        target = torch.eye(num_class)[target.cpu()].tolist()
        for i in range(len(target)):
            total_pearson += pearson(target[i], output[i].cpu().detach().tolist())
        total_num += len(target)
    val_acc = total_pearson / total_num
    print(name + ' Val Pearson: {:.6f}'.format(val_acc))

    # total_num = 0
    # total_pearson = 0
    # for idx, mask, target in test_dataloader:
    #     idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
    #     output, cls_result, type_id, shared, specific = model(idx, mask, name)
    #     target = torch.eye(target.size(1))[target]
    #     for i in range(target.size(0)):
    #         total_pearson += pearsonr(target[i], output[i]).statistic
    #     total_num += target.size(0)
    # test_acc = total_pearson / total_num
    # print(name + ' Test Pearson: {:.6f}'.format(test_acc))
    return [val_acc]


def train_datasets(name, dataset_loader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch):
    for batch_idx, (idx, mask, target) in enumerate(dataset_loader):
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output, cls_result, type_id, shared, specific = model(idx, mask, name)
        optimizer.zero_grad()
        loss = Loss(output, target)
        if use_adv:
            loss += alpha * Loss(cls_result, (torch.ones_like(target) * type_id).long().to(device))
        if use_diff:
            loss += beta * Diff_Loss(shared, specific)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dataset_loader), 100. * (batch_idx + 1) / len(dataset_loader),
                loss.item()))

def train_transfer(name, dataset_loader, device, model, optimizer, Loss, epoch):
    for batch_idx, (idx, mask, target) in enumerate(dataset_loader):
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output  = model(idx, mask, name)
        optimizer.zero_grad()
        loss = Loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dataset_loader), 100. * (batch_idx + 1) / len(dataset_loader),
                loss.item()))

def test_transfer(train_dataloader, val_dataloader, test_dataloader, device, model, name):
    correct = 0
    total_num = 0
    for idx, mask, target in val_dataloader:
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output = model(idx, mask, name)
        pred = output.argmax(dim=1)
        correct += int(pred.eq(target).sum())
        total_num += pred.size(0)
    val_acc = correct / total_num
    print(name + ' Val Accuracy: {:.6f}'.format(correct / total_num))

    correct = 0
    total_num = 0
    for idx, mask, target in test_dataloader:
        idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        output = model(idx, mask, name)
        pred = output.argmax(dim=1)
        correct += int(pred.eq(target).sum())
        total_num += pred.size(0)
    test_acc = correct / len(test_dataloader)
    print(name + ' Test Accuracy: {:.6f}'.format(correct / total_num))
    return [val_acc, test_acc]



def train(datasets_names=None, lr=5e-5, optimizer="AdamW", batch_size=8, model_name='bert-base-uncased', use_adv=True,
          use_diff=True, alpha=0.05, beta=0.01, model_init=None, train_epoch=10, need_transfer=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Bert(model_name=model_name, dataset_names=datasets_names, need_transfer_test=need_transfer).to(device)
    if model_init != None:
        model.load_state_dict(torch.load(model_init))
    if optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    Loss = nn.CrossEntropyLoss()
    Diff_Loss = diff_loss()
    if datasets_names==None or "mrpc" in datasets_names:
        mrpc_train_dataloader, mrpc_val_dataloader, mrpc_test_dataloader = Get_datasets("mrpc", batch_size=batch_size)
        mrpc_acc = []
    if datasets_names==None or "stsb" in datasets_names:
        stsb_train_dataloader, stsb_val_dataloader, stsb_test_dataloader = Get_datasets("stsb", batch_size=batch_size)
        stsb_acc = []
    if datasets_names == None or "sst2" in datasets_names:
        sst2_train_dataloader, sst2_val_dataloader, sst2_test_dataloader = Get_datasets("sst2", batch_size=batch_size)
        sst2_acc = []
    if datasets_names == None or "qnli" in datasets_names:
        qnli_train_dataloader, qnli_val_dataloader, qnli_test_dataloader = Get_datasets("qnli", batch_size=batch_size)
        qnli_acc = []
    if datasets_names == None or "cola" in datasets_names:
        cola_train_dataloader, cola_val_dataloader, cola_test_dataloader = Get_datasets("cola", batch_size=batch_size)
        cola_acc = []
    if datasets_names == None or "ax" in datasets_names:
        ax_train_dataloader, ax_val_dataloader, ax_test_dataloader = Get_datasets("ax", batch_size=batch_size)
        ax_acc = []
    if datasets_names == None or "mnli" in datasets_names:
        mnli_train_dataloader, mnli_val_dataloader, mnli_test_dataloader = Get_datasets("mnli", batch_size=batch_size)
        mnli_acc = []
    if datasets_names == None or "wnli" in datasets_names:
        wnli_train_dataloader, wnli_val_dataloader, wnli_test_dataloader = Get_datasets("wnli", batch_size=batch_size)
        wnli_acc = []
    if datasets_names == None or "qqp" in datasets_names:
        qqp_train_dataloader, qqp_val_dataloader, qqp_test_dataloader = Get_datasets("qqp", batch_size=batch_size)
        qqp_acc = []
    if datasets_names == None or "rte" in datasets_names:
        rte_train_dataloader, rte_val_dataloader, rte_test_dataloader = Get_datasets("rte", batch_size=batch_size)
        rte_acc = []
    if need_transfer:
        srte_train_dataloader, srte_val_dataloader, srte_test_dataloader = Get_datasets("srte", batch_size=batch_size)
    # if datasets_names==None or "onto" in datasets_names:
    #     onto_train_dataloader, onto_val_dataloader, onto_test_dataloader = Get_datasets("onto", batch_size=batch_size)
    # if datasets_names == None or "wmt19" in datasets_names:
    #     wmt_train_dataloader, wmt_val_dataloader, wmt_test_dataloader = Get_datasets("wmt19", batch_size=batch_size)
    for epoch in range(train_epoch):
        if datasets_names == None or "mrpc" in datasets_names:
            train_datasets("mrpc", mrpc_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            mrpc_list = val_and_test(mrpc_val_dataloader, mrpc_test_dataloader, device, model, "mrpc")
            mrpc_acc.append(mrpc_list)
        if datasets_names == None or "stsb" in datasets_names:
            train_datasets("stsb", stsb_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            stsb_list = val_and_test_pearson(stsb_val_dataloader, stsb_test_dataloader, device, model, "stsb", 6)
            stsb_acc.append(stsb_list)
        if datasets_names == None or "sst2" in datasets_names:
            train_datasets("sst2", sst2_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            sst2_list = val_and_test(sst2_val_dataloader, sst2_test_dataloader, device, model, "sst2")
            sst2_acc.append(sst2_list)
        if datasets_names == None or "qnli" in datasets_names:
            train_datasets("qnli", qnli_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            qnli_list = val_and_test(qnli_val_dataloader, qnli_test_dataloader, device, model, "qnli")
            qnli_acc.append(qnli_list)
        if datasets_names == None or "cola" in datasets_names:
            train_datasets("cola", cola_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            cola_list = val_and_test_mcc(cola_val_dataloader, cola_test_dataloader, device, model, "cola")
            cola_acc.append(cola_list)
        if datasets_names == None or "ax" in datasets_names:
            train_datasets("ax", cola_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            ax_list = val_and_test(ax_val_dataloader, ax_test_dataloader, device, model, "ax")
            ax_acc.append(ax_list)
        if datasets_names == None or "mnli" in datasets_names:
            train_datasets("mnli", mnli_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            mnli_list = val_and_test(mnli_val_dataloader, mnli_test_dataloader, device, model, "mnli")
            mnli_acc.append(mnli_list)
        if datasets_names == None or "wnli" in datasets_names:
            train_datasets("wnli", wnli_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            wnli_list = val_and_test(wnli_val_dataloader, wnli_test_dataloader, device, model, "wnli")
            wnli_acc.append(wnli_list)
        if datasets_names == None or "qqp" in datasets_names:
            train_datasets("qqp", qqp_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            qqp_list = val_and_test(qqp_val_dataloader, qqp_test_dataloader, device, model, "qqp")
            qqp_acc.append(qqp_list)
        if datasets_names == None or "rte" in datasets_names:
            train_datasets("rte", rte_train_dataloader, device, model, optimizer, Loss, use_adv, use_diff, Diff_Loss, alpha, beta, epoch)
            rte_list = val_and_test(rte_val_dataloader, rte_test_dataloader, device, model, "rte")
            rte_acc.append(rte_list)
        # if datasets_names == None or "onto" in datasets_names:
        #     for batch_idx, (idx, mask, target) in enumerate(onto_train_dataloader):
        #         idx, mask, target = idx.to(device), mask.to(device), target.to(device).long()
        #         crf_loss, train_result, cls_result, type_id, shared, specific = model(idx, mask, "onto", labels=target)
        #         optimizer.zero_grad()
        #         loss = crf_loss
        #         if use_adv:
        #             loss += alpha * Loss(cls_result, (torch.flatten(torch.ones_like(target)) * type_id).long().to(device))
        #         loss.backward()
        #         optimizer.step()
        #         if (batch_idx + 1) % 50 == 0:
        #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 epoch, batch_idx + 1, len(onto_train_dataloader), 100. * (batch_idx + 1) / len(onto_train_dataloader),
        #                 loss.item()))
        # if datasets_names == None or "wmt19" in datasets_names:
        #     for batch_idx, data in enumerate(wmt_train_dataloader):
        #         idx = data['input_ids'].to(device)
        #         mask = data['attention_mask'].to(device)
        #         decoder_idx = data['decoder_input_ids'].to(device)
        #         labels = data['labels'].to(device)
        #         idx, mask, decoder_idx = idx.to(device), mask.to(device), decoder_idx.to(device)
        #         logits, cls_type, type_id, clf_feature_shared, out_wmt = model(idx, mask, "wmt19", decoder_input_ids=decoder_idx)
        #         optimizer.zero_grad()
        #         loss = Loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        #         if use_adv:
        #             loss += alpha * Loss(cls_result, (torch.flatten(torch.ones_like(target)) * type_id).long().to(device))
        #         loss.backward()
        #         optimizer.step()
        #         if (batch_idx + 1) % 50 == 0:
        #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 epoch, batch_idx + 1, len(wmt_train_dataloader), 100. * (batch_idx + 1) / len(wmt_train_dataloader),
        #                 loss.item()))

        # if datasets_names == None or "mrpc" in datasets_names:
        #     mrpc_list = val_and_test(mrpc_val_dataloader, mrpc_test_dataloader, device, model, "mrpc")
        #     mrpc_acc.append(mrpc_list)
        # if datasets_names == None or "stsb" in datasets_names:
        #     stsb_list = val_and_test(stsb_val_dataloader, stsb_test_dataloader, device, model, "stsb")
        #     stsb_acc.append(stsb_list)
        # if datasets_names == None or "qnli" in datasets_names:
        #     qnli_list = val_and_test(qnli_val_dataloader, qnli_test_dataloader, device, model, "qnli")
        #     qnli_acc.append(qnli_list)
        # if datasets_names == None or "sst2" in datasets_names:
        #     sst2_list = val_and_test(sst2_val_dataloader, sst2_test_dataloader, device, model, "sst2")
        #     sst2_acc.append(sst2_list)
        # if datasets_names == None or "cola" in datasets_names:
        #     cola_list = val_and_test(cola_val_dataloader, cola_test_dataloader, device, model, "cola")
        #     cola_acc.append(cola_list)
        # if datasets_names == None or "ax" in datasets_names:
        #     ax_list = val_and_test(ax_val_dataloader, ax_test_dataloader, device, model, "ax")
        #     ax_acc.append(ax_list)
        # if datasets_names == None or "mnli" in datasets_names:
        #     mnli_list = val_and_test(mnli_val_dataloader, mnli_test_dataloader, device, model, "mnli")
        #     mnli_acc.append(mnli_list)
        # if datasets_names == None or "wnli" in datasets_names:
        #     wnli_list = val_and_test(wnli_val_dataloader, wnli_test_dataloader, device, model, "wnli")
        #     wnli_acc.append(wnli_list)
        # if datasets_names == None or "qqp" in datasets_names:
        #     qqp_list = val_and_test(qqp_val_dataloader, qqp_test_dataloader, device, model, "qqp")
        #     qqp_acc.append(qqp_list)
        # if datasets_names == None or "rte" in datasets_names:
        #     rte_list = val_and_test(rte_val_dataloader, rte_test_dataloader, device, model, "rte")
        #     rte_acc.append(rte_list)
        # torch.save(model.state_dict(), "./checkpoint/model_parameter_for_epoch"+str(epoch)+".pkl")
    return_dic = {}
    if datasets_names == None or "mrpc" in datasets_names:
        return_dic["mrpc"] = mrpc_acc
    if datasets_names == None or "stsb" in datasets_names:
        return_dic["stsb"] = stsb_acc
    if datasets_names == None or "qnli" in datasets_names:
        return_dic["qnli"] = qnli_acc
    if datasets_names == None or "sst2" in datasets_names:
        return_dic["sst2"] = sst2_acc
    if datasets_names == None or "cola" in datasets_names:
        return_dic["cola"] = cola_acc
    if datasets_names == None or "ax" in datasets_names:
        return_dic["ax"] = ax_acc
    if datasets_names == None or "mnli" in datasets_names:
        return_dic["mnli"] = mnli_acc
    if datasets_names == None or "qqp" in datasets_names:
        return_dic["qqp"] = qqp_acc
    if datasets_names == None or "rte" in datasets_names:
        return_dic["rte"] = rte_acc

    if need_transfer:
        for epoch in range(train_epoch):
            train_transfer("srte", srte_train_dataloader, device, model, optimizer, Loss, epoch)
            srte_list = test_transfer(srte_train_dataloader, srte_val_dataloader, srte_test_dataloader, device, model, "srte")

    return return_dic


if __name__ == '__main__':
    args = parse_args()
    datasets = []
    datasets_names = args.datasets
    if datasets_names == "all":
        datasets = None
    else:
        datasets = datasets_names.split(",")
    train(datasets, batch_size=args.batch_size, use_adv=args.use_adv, use_diff=args.use_diff,
          need_transfer=args.need_transfer, train_epoch=args.train_epoch, lr=args.lr)
