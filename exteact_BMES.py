# for extract label and other information from origin .bmes file

import os
import pandas as pd
from preprocess import preproces

anns_dir = "../labeled_set/"
entityRe = r'\[\@.*?\#.*?\*\](?!\#)'
title_list = ["doc_id", "doc_name", "par_id", "par", "label", "par_length"]

result = []
dir = os.listdir(anns_dir)

"""
d = 0
for file in dir:
    if file[-4:] == ".ann":
        with open(anns_dir+file, "r", encoding="utf8") as f:
            document = f.read()
        paragraph = document.split("\n")
        p = 0
        for par in paragraph:
            if len(par.strip()) > 0:
                entity = re.search(entityRe, par)
                if entity:
                    entity_pair = preproces.getLabelPair(entity.group())
                    result.append([d, file, p, entity_pair["text"], entity_pair["tag"]])
                else:
                    result.append([d, file, p, par, "Other"])
                p = p + 1

        d = d + 1
pd.DataFrame(result, columns=title_list).to_csv("extract_label/data.csv", header=True)
"""

d = 0
for file in dir:
    if file[-4:] == ".ann":
        with open(anns_dir+file, "r", encoding="utf8") as f:
            document = f.read()
        par_pair = preproces.MergeFollows(document, isBMES=True).mergeBMESPair
        for p, item in enumerate(par_pair):
            if len(item["par"].strip()) > 0:
                result.append([d, file.replace("grj.", "").replace(".ann", ""), p, item["par"], item["label"], len(item["par"])])
    d = d + 1
pd.DataFrame(result, columns=title_list).to_csv("../dataset/hot_app_pp.csv", header=True)



