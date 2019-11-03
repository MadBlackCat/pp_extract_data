import re
import json
import math
import pandas as pd
urlReg = r"^(?:([A-Za-z]+):)?(\/{0,3})([0-9.\-A-Za-z]+)(?::(\d+))?(?:\/([^?#]*))?(?:\?([^#]*))?(?:#(.*))?$"
entityRe = r'\[\@.*?\#.*?\*\](?!\#)'


def getLabelPair(text):
    label_pair = re.search(entityRe, text)
    if label_pair:
        new_string_list = label_pair.group().strip('[@*]').rsplit('#', 1)
        par_text = new_string_list[0]
        label = new_string_list[1]
        label = "Legal Basis" if label.lower() == "legal basis" else label
        label = "User Control" if label == "User Choice/Control" else label
        sentiment = 0
    else:
        par_text = text
        label = "Other"
        sentiment = 0
    return {"text": par_text, "label": label, "sentiment" : sentiment}


def removeLabel(text):
    return text.replace("[@", "").rsplit('#', 1)[0]


class MergeFollows:
    def __init__(self, document, isBMES = False):
        self.text = document
        self.isBMES = isBMES
        self._start = "start"
        self._unTAG = "unTAG"
        self._item = "item"
        self.par_list = self.text.split('\n')
        if isBMES:
            self.par_dict = [{"par": getLabelPair(par)["text"],
                              "tag": self._unTAG,
                              "label": getLabelPair(par)["label"]}
                             for par in self.par_list]

        else:
            self.par_dict = [{"par": par, "tag": self._unTAG} for par in self.par_list]
        self._merged_par = []
        self.theEndofStr = ("or", "and", ";")
        self._initDict()

    def _initDict(self):
        for key, item in enumerate(self.par_dict):
            text = item["par"].strip()
            last = self._getLastItem(key)
            next = self._getNextItem(key)
            #  as follows: check if have :
            if self._isStart(text):
                self.par_dict[key]["tag"] = self._start
            # after :, the first str must be the item
            elif key != 0 and self.par_dict[key-1]["tag"] == self._start:
                self.par_dict[key]["tag"] = self._item
            # ignore the \n between start and item and set the first str (after : and \n) is item
            elif self._maybeItem(key, notNULL=False) and len(last) == 0:
                self.par_dict[key]["tag"] = self._item
                if key + 1 != len(self.par_dict):
                    self.par_dict[key+1]["tag"] = self._item
            # if the fist char is special char
            elif self._maybeItem(key) and not text[0].isalnum():
                self.par_dict[key]["tag"] = self._item
            elif self._maybeItem(key, notNULL=False) and not last[0].isalnum() and last[0] == next[0]:
                self.par_dict[key]["tag"] = self._item
            # such as "to access  your personal information."
            elif self._maybeItem(key) and text[:2].lower() == "to ":
                self.par_dict[key]["tag"] = self._item
            elif self._maybeItem(key, notNULL=False) and last[:2].lower() == "to " and last[:2] == next[:2]:
                self.par_dict[key]["tag"] = self._item
            # such as "Request a structured electronic version of your information;"
            # or "Request a structured electronic version of your information; and"
            # Besides set the next str to the item
            elif self._maybeItem(key) and text.endswith(self.theEndofStr):
                self.par_dict[key]["tag"] = self._item
                if key + 1 != len(self.par_dict):
                    self.par_dict[key + 1]["tag"] = self._item
            # the last item is start with number end this item is also start with item
            elif self._maybeItem(key) and text.strip()[0].isdigit() and self.par_dict[key - 1]["par"][0].isdigit():
                self.par_dict[key]["tag"] = self._item
            # the link of third party
            elif self._maybeItem(key) and self._calUrl(text) > 0.5:
                self.par_dict[key]["tag"] = self._item
            else:
                pass

    def _getLastItem(self, key):
        return self.par_dict[key - 1]["par"].strip() \
            if key != 0 and len(self.par_dict[key - 1]["par"].strip()) > 0 else "null"

    def _getNextItem(self, key):
        return self.par_dict[key + 1]["par"].strip() \
            if key+1 != len(self.par_dict) and len(self.par_dict[key + 1]["par"].strip()) else "null"

    def _maybeItem(self, key, notNULL=True):
        if notNULL:
            maybe = key != 0 and self.par_dict[key - 1]["tag"] != self._unTAG and len(self.par_dict[key]["par"].strip()) > 0
        else:
            maybe = key != 0 and self.par_dict[key - 1]["tag"] != self._unTAG and len(self.par_dict[key]["par"].strip()) == 0
        return maybe

    @staticmethod
    def _isStart(paragraph):
        return True if len(paragraph.strip()) > 0 and paragraph.strip()[-1] == ":" else False

    @staticmethod
    def _calUrl(text):
        max_url_ken = 0
        if len(text.strip()) > 0:
            for i in text.split(" "):
                url = re.search(urlReg, i)
                if url and len(url.group()) / len(text) > max_url_ken:
                    max_url_ken = len(url.group()) / len(text)
        return max_url_ken

    @property
    def mergeBMESPair(self):
        if self.isBMES:
            for key, item in enumerate(self.par_dict):
                if item["tag"] == self._unTAG or item["tag"] == self._start:
                    self._merged_par.append({"par": item["par"], "label": item["label"]})
                else:
                    self._merged_par[-1]["par"] += item["par"]
        else:
            for key, item in enumerate(self.par_dict):
                if item["tag"] == self._unTAG or item["tag"] == self._start:
                    self._merged_par.append({"par": item["par"], "label": "None"})
                else:
                    self._merged_par[-1]["par"] += item["par"]
        return self._merged_par

    @property
    def merge(self):
        for key, item in enumerate(self.par_dict):
            if item["tag"] == self._unTAG or item["tag"] == self._start:
                self._merged_par.append(item["par"])
            else:
                self._merged_par[-1] += item["par"]
        return self._merged_par






class CalDataLabel():
    def __init__(self, data_path, label_path, dev_split = 0.1, test_split = 0.1):
        self.data = pd.read_csv(data_path)

        with open(label_path, "r", encoding="utf8") as f:
            self.label_dict = json.load(f)
        self.dev_split = dev_split
        self.test_split = test_split
        self.label_group = self.data[self.data.label != "Other"].groupby("label")
        self.doc_group = self.data.groupby("doc_id")

        self.label_size = {label: size for label, size in self.label_group.size().iteritems()}

    def cal_words(self, out_path):
        pd.DataFrame(self.data[self.data.label != "Other"].par_length.describe()).to_csv(out_path+"data_status.csv")

        pd.DataFrame(self.label_group.par_length.describe()).to_csv(out_path+"label_status.csv")
        pd.DataFrame(self.label_group.size()).to_csv(out_path+"label_size.csv")
        pd.DataFrame(self.doc_group.par_length.sum().describe()).to_csv(out_path+"doc_status.csv")
        # pd.concat([label_status, par_status, len_status], axis=1).to_csv(out_path+"data_status.csv")

    def extract_all_label(self):
        label_set = set()
        for key, item in self.label_dict.items():
            label_set.add(item)
        for i in label_set:
            bi_label_dict = dict()
            for label in self.label_dict.keys():
                if self.label_dict[label] != i:
                    bi_label_dict[label] = 0
                else:
                    bi_label_dict[label] = 1
            with open("./config/"+str(i)+".json", "w", encoding="utf-8") as fin:
                fin.write(json.dumps(bi_label_dict))

    def extract_data(self, out_path, all_data=None, down_sampling=True):
        if all_data is None:
            data = self.data[self.data.label != "Other"]
        else:
            data = self.data

        document_index = data.doc_id.drop_duplicates()
        # print(document)
        dev_test_index = document_index.sample(frac=self.dev_split + self.test_split, random_state=666)
        train_index = document_index[~document_index.index.isin(dev_test_index.index)]

        train = data[data.doc_id.isin(train_index)].sample(frac=1, random_state=666)
        dev_test = data[data.doc_id.isin(dev_test_index)].sample(frac=1, random_state=666)

        test_index = dev_test_index.sample(frac=self.test_split / (self.dev_split + self.test_split), random_state=666)
        dev_index = dev_test_index[~dev_test_index.index.isin(test_index.index)]

        dev = dev_test[dev_test.doc_id.isin(dev_index)].sample(frac=1, random_state=666)
        test = dev_test[dev_test.doc_id.isin(test_index)].sample(frac=1, random_state=666)

        train_data = []
        dev_data = []
        test_data = []

        for i, row in train.iterrows():
            train_data.append([self.label_dict[row["label"]], row["par"]])
        for i, row in dev.iterrows():
            dev_data.append([self.label_dict[row["label"]], row["par"]])
        for i, row in test.iterrows():
            test_data.append([self.label_dict[row["label"]], row["par"]])
        train_data = pd.DataFrame(train_data, columns=["sentiment","review"])
        dev_data = pd.DataFrame(dev_data, columns=["sentiment","review"])
        test_data = pd.DataFrame(test_data, columns=["sentiment","review"])

        if down_sampling:
            is_label = train_data[train_data.sentiment == 1]
            no_label = train_data[train_data.sentiment != 1].sample(n=len(is_label))

            fin_train_data = pd.concat([is_label, no_label]).sample(frac=1, random_state=666)
        else:
            fin_train_data = train_data
        print("Train Data Num - " + str(len(fin_train_data)), " ||| Dev Data Num - " + str(len(dev_data)), " ||| Test Data - " + str(len(test_data)))
        fin_train_data.to_csv(out_path+"_train_data.tsv", sep="\t")
        dev_data.to_csv(out_path+"_dev_data.tsv", sep="\t")
        test_data.to_csv(out_path+"_test_data.tsv", sep="\t")

    def extract_bi_data(self, out_path):
        data = self.data[self.data.label != "Other"]
        classification_size = 0
        classification_label = []
        other_label = []
        for label, size in self.label_size.items():
            if self.label_dict[label] == 1:
                classification_size += size
                classification_label.append(label)
            else:
                other_label.append(label)
        split_size = (classification_size / len(other_label))
        title_list = ["doc_id", "doc_name", "par_id", "par", "label", "par_length"]
        other_data = pd.DataFrame(columns=title_list)

        classification_data = data[data.label.isin(classification_label)]
        for label in other_label:
            label_data = data[data.label == label]
            if len(label_data) + 10 <= split_size:
                # other_data.append(label_data.sample(frac=1, random_state=666))
                if len(other_data) > 0:
                    other_data = pd.concat([other_data, label_data.sample(frac=1, random_state=666)])
                else:
                    other_data = label_data.sample(frac=1, random_state=666)
            else:
                # other_data.append(label_data.sample(n=math.ceil(split_size)+10, random_state=666))
                if len(other_data) > 0:
                    other_data = pd.concat([other_data, label_data.sample(n=math.ceil(split_size), random_state=666)])
                else:
                    other_data = label_data.sample(n=math.ceil(split_size), random_state=666)

        all_data = pd.concat([classification_data, other_data])
        self.extract_data(out_path, all_data)
