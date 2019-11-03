# extract label from .csv file

from preprocess import preproces
import json

orgin_path = "../dataset/app_pp_label.csv"

# data = preproces.CalDataLabel(orgin_path).cal_label("../data_status/")

#
# result=[[label_dict[item["label"]], item["par"]] for index, item in data.iterrows()]
# pd.DataFrame(result).to_csv("../dataset/train.tsv", sep="\t")
#
#

with open("./config/label.json", "r", encoding="utf8") as f:
    label_dict = json.load(f)

label = [key for key, item in label_dict.items()]
for key in label:
    classification_name = key.replace(" ", "_").replace("/", "_")
#     # dict = {label_name: 0 for label_name in label}
#     # dict[key] = 1
#     # with open("../config/"+classification_name+".json", "w", encoding="utf8") as f:
#     #     json.dump(dict, f)
#     # classification_name = "data_retention"
#     classification_name = "15_classification"
#     preproces.CalDataLabel(orgin_path, "../config/"+classification_name+".json").extract_data("./data/" + classification_name, bi_label=classification_name)
for i in range(0,10):
    classification_name = str(i)
    preproces.CalDataLabel(orgin_path, "./config/"+classification_name+".json").extract_data("./un_downsampling/" + classification_name, down_sampling=False)
