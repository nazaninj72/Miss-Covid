from utils import *
from fc import FactCheck
import pandas as pd 

def polify(val):
    val = val.replace(",","")
    politopic=""
    first=True
    for w in val.split(" "):
        if first:
            politopic+=w
            first=False
        else:
            politopic+="+"+w
    return politopic


def get_key(val,my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
            return key 
  
    return "key doesn't exist"     

# f = open("resources/wikitopics.txt", "r") 
# lines = f.readlines()
# wikitopics=dict()
# wikisubtopics=dict()
# for line in lines:
#     topiclevel=line.split("\t")[0]
#     topicname=line.split("\t")[1].replace("\n","")
#     if topiclevel.find(".") == -1:
#         wikitopics[topiclevel]=topicname
# for line in lines:
#     topiclevel=line.split("\t")[0]
#     topicname=line.split("\t")[1].replace("\n","")
#     print(topiclevel.find("."))
#     if topiclevel.find(".") > -1:
#         wikisubtopics[wikitopics[topiclevel.split(".")[0]]].append([topicname])
#     else:
#         wikisubtopics[wikitopics[topiclevel]]=[""]


politopic=""
polisubtopic=""
fc=FactCheck()
records=[]
import json,sys,csv

politopic="common flu"
# for val in wikitopics.values():

#     if len(wikisubtopics[val])== 1:
#         polisubtopic=""
#         politopic=polify(val)
       
#         print(politopic)
politopic=polify(politopic)
records=fc.crawl_politifact(politopic,polisubtopic)
with open('polifacts-flu.csv', 'w', newline='') as myfile:
    mycsv.writerow(['source','date','content','url', 'Label'])
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(records)
    # else:
    #     for subtopic in wikisubtopics[val]:
    #         if subtopic=="":
    #             continue
    #         else:
    #             polisubtopic=polify(subtopic[0])
    #             politopic=polify(val)
    #             fc.crawl_politifact(politopic,polisubtopic,mycsv)


