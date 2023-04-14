import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertModel
from datasets import load_dataset


def get_token_and_pretrained(model_name, need_token=True, need_pretrained=True):
    token = None
    pretrained = None
    if need_token:
        token = BertTokenizer.from_pretrained(model_name)
    if need_pretrained:
        pretrained = BertModel.from_pretrained(model_name)
    return token, pretrained


class TwoSentenceDataset():
    def __init__(self, TwoSentenceDataset, token, first='sentence1', second='sentence2', label="label"):
        data = token(TwoSentenceDataset[first], TwoSentenceDataset[second],
                     truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        self.input_ids, self.attention_masks = data.input_ids, data.attention_mask
        self.labels = torch.Tensor(TwoSentenceDataset[label])

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        attention_mask = self.attention_masks[index]
        label = self.labels[index]
        return input_id, attention_mask, label

    def __len__(self):
        return len(self.labels)

class SingleSentenceDataset():
    def __init__(self, dataset, token, first="sentence", label="label"):
        data = token(dataset[first], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        self.input_ids, self.attention_masks = data.input_ids, data.attention_mask
        self.labels = torch.Tensor(dataset[label])

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        attention_mask = self.attention_masks[index]
        label = self.labels[index]
        return input_id, attention_mask, label

    def __len__(self):
        return len(self.labels)


class nerDataset():
    def __init__(self, nerdataset, token, length=None):
        if length == None:
            length = len(nerdataset)
        for i in range(length):
            nerdataset["tokens"][i] = " ".join(nerdataset["tokens"][i])
        data = token(nerdataset['tokens'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        self.input_ids, self.attention_masks = data.input_ids, data.attention_mask
        self.labels = torch.Tensor(self.load_data(nerdataset['tags'], max_len=128))

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        attention_mask = self.attention_masks[index]
        label = self.labels[index]
        return input_id, attention_mask, label

    def __len__(self):
        return len(self.labels)

    def load_data(self, labels=None, max_len=128):
        y = []
        if labels:
            for i in np.arange(len(labels)):
                if len(labels[i]) < max_len:
                    y.append(labels[i] + [0]*(max_len - len(labels[i])))
                else:
                    y.append(labels[i][:max_len])
        return np.asarray(y)

class TranslationDataset():
    def __init__(self, datasets, tokenizer, length=None):
        self.length = length
        self.datasets = datasets
        self.tokenizer = tokenizer

    def __len__(self):
        if self.length == None:
            return len(self.datasets)
        else:
            return self.length

    def __getitem__(self, idx):
        src_text = self.datasets[idx]["en"]
        tgt_text = self.datasets[idx]["zh"]
        src_encoding = self.tokenizer(src_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        tgt_encoding = self.tokenizer(tgt_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {'input_ids': src_encoding['input_ids'].squeeze(), 'attention_mask': src_encoding['attention_mask'].squeeze(),
                'decoder_input_ids': tgt_encoding['input_ids'].squeeze()[:-1], 'decoder_attention_mask': tgt_encoding['attention_mask'].squeeze()[:-1],
                'labels': tgt_encoding['input_ids'].squeeze()[1:]}

def get_two_sentence_datasets(name, token_name="bert-base-uncased", batch_size=8, shuffle=True, first='sentence1', second='sentence2', label="label"):
    if name == "srte":
        token, _ = get_token_and_pretrained(token_name, need_pretrained=False)
        raw_datasets = load_dataset("super_glue", "rte")
        Train = TwoSentenceDataset(raw_datasets["train"], token, first=first, second=second, label=label)
        Val = TwoSentenceDataset(raw_datasets["validation"], token, first=first, second=second, label=label)
        Test = TwoSentenceDataset(raw_datasets["test"], token, first=first, second=second, label=label)
        train_dataloader = torch.utils.data.DataLoader(Train, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(Val, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = torch.utils.data.DataLoader(Test, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, val_dataloader, test_dataloader
    elif name in ["mrpc", "stsb", "qnli", "rte", "qqp", "wnli", "mnli", "ax"]:
        token, _ = get_token_and_pretrained(token_name, need_pretrained=False)
        raw_datasets = load_dataset("glue", name)
        if name == "mnli":
            Train = TwoSentenceDataset(raw_datasets["train"], token, first=first, second=second, label=label)
            Val = TwoSentenceDataset(raw_datasets["validation_matched"], token, first=first, second=second, label=label)
            Test = TwoSentenceDataset(raw_datasets["test_matched"], token, first=first, second=second, label=label)
        else:
            Train = TwoSentenceDataset(raw_datasets["train"], token, first=first, second=second, label=label)
            Val = TwoSentenceDataset(raw_datasets["validation"], token, first=first, second=second, label=label)
            Test = TwoSentenceDataset(raw_datasets["test"], token, first=first, second=second, label=label)
        train_dataloader = torch.utils.data.DataLoader(Train, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(Val, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = torch.utils.data.DataLoader(Test, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, val_dataloader, test_dataloader

def get_single_sentence_datasets(name, token_name="bert-base-uncased", batch_size=8, shuffle=True, first='sentence', label="label"):
    if name in ["sst2", "cola"]:
        token, _ = get_token_and_pretrained(token_name, need_pretrained=False)
        raw_datasets = load_dataset("glue", name)
        Train = SingleSentenceDataset(raw_datasets["train"], token, first=first, label=label)
        Val = SingleSentenceDataset(raw_datasets["validation"], token, first=first, label=label)
        Test = SingleSentenceDataset(raw_datasets["test"], token, first=first, label=label)
        train_dataloader = torch.utils.data.DataLoader(Train, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(Val, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = torch.utils.data.DataLoader(Test, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, val_dataloader, test_dataloader


def Get_datasets(name, batch_size=8, shuffle=True, token_name="bert-base-uncased", train_num=None):
    if name == "mrpc":
        return get_two_sentence_datasets("mrpc", batch_size=batch_size, shuffle=shuffle)
    elif name == "stsb":
        return get_two_sentence_datasets("stsb", batch_size=batch_size, shuffle=shuffle)
    elif name == "wnli":
        return get_two_sentence_datasets("wnli", batch_size=batch_size, shuffle=shuffle)
    elif name == "qqp":
        return get_two_sentence_datasets("qqp", batch_size=batch_size, shuffle=shuffle, first="question1", second="question2")
    elif name == "qnli":
        return get_two_sentence_datasets("qnli", batch_size=batch_size, shuffle=shuffle, first="question", second="sentence")
    elif name == "rte":
        return get_two_sentence_datasets("rte", batch_size=batch_size, shuffle=shuffle)
    elif name == "mnli":
        return get_two_sentence_datasets("mnli", batch_size=batch_size, shuffle=shuffle, first="premise", second="hypothesis")
    elif name == "cola":
        return get_single_sentence_datasets("cola", batch_size=batch_size, shuffle=shuffle)
    elif name == "sst2":
        return get_single_sentence_datasets("sst2", batch_size=batch_size, shuffle=shuffle)
    elif name == "ax":
        return get_two_sentence_datasets("ax", batch_size=batch_size, shuffle=shuffle, first="premise", second="hypothesis")
    elif name == "srte":
        return get_two_sentence_datasets("srte", batch_size=batch_size, shuffle=shuffle, first="premise", second="hypothesis")
    elif name == "onto":
        token, _ = get_token_and_pretrained(token_name, need_pretrained=False)
        raw_datasets = load_dataset("tner/ontonotes5")
        if train_num==None:
            train_num = len(raw_datasets["train"])
        ontoTrain = nerDataset(raw_datasets["train"], token, train_num)
        ontoVal = nerDataset(raw_datasets["validation"], token)
        ontoTest = nerDataset(raw_datasets["test"], token)
        train_dataloader = torch.utils.data.DataLoader(ontoTrain, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(ontoVal, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = torch.utils.data.DataLoader(ontoTest, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, val_dataloader, test_dataloader
    elif name == "wmt19":
        token, _ = get_token_and_pretrained(token_name, need_pretrained=False)
        raw_datasets = load_dataset("wmt19", "zh-en")
        if train_num==None:
            train_num = len(raw_datasets["train"])
        wmtTrain = TranslationDataset(raw_datasets["train"], token, train_num)
        wmtVal = TranslationDataset(raw_datasets["validation"], token)
        wmtTest = TranslationDataset(raw_datasets["test"], token)
        train_dataloader = torch.utils.data.DataLoader(wmtTrain, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(wmtVal, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = torch.utils.data.DataLoader(wmtTest, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, val_dataloader, test_dataloader

    return None, None, None

