# -*- coding: utf-8 -*-
# 用于将文本中的实体概念与微软KG中的概念实体进行映射的工具
import pandas as pd
import tagme
import logging
import sys
import os.path
import requests
import json
from tqdm import tqdm

# 标注的“Authorization Token”，需要注册才有
tagme.GCUBE_TOKEN = "d866f962-a8f3-4213-a93b-fc0c1383a973-843339462"
# 获取当前程序的名称
program = os.path.basename(sys.argv[0])
# 设置日志格式
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

# 返回带有所有概念实体的字典
def get_instance_concept(file):
    '''
    返回一个包含键值对的字典，其中键为实体概念，值为至少一个与实体概念相应的维基百科中的所有概念。
    函数从给定文件中逐行读取数据，将实体概念和对应的概念实体添加到字典中。
    如果给定实体概念不在字典中，则将其添加到字典中并将其值初始化为空列表。
    如果给定实体概念已存在于字典中，则将概念实体添加到概念实体列表中。
    函数最终返回带有所有概念实体的字典。
    '''
    ent_concept = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            cpt = line[0]
            ent = line[1]
            if ent not in ent_concept:
                ent_concept[ent] = []
            ent_concept[ent].append(cpt)

    return ent_concept

def Annotation_mentions(txt):
    """
    发现那些文本中可以是维基概念实体的概念
    :param txt: 一段文本对象，str类型
    :return: 键值对，键为本文当中原有的实体概念，值为该概念作为维基概念的概念大小，那些属于维基概念但是存在歧义现象的也包含其内
    """
    # 使用TagMe API进行文本实体标注
    annotation_mentions = tagme.mentions(txt)
    # 创建一个空字典
    dic = dict()
    # 遍历文本中的所有实体
    for mention in annotation_mentions.mentions:
        try:
            # 将实体概念和对应的概念大小添加到字典中
            dic[str(mention).split(" [")[0]] = str(mention).split("] lp=")[1]
        except:
            # 如果出现异常，则记录日志
            logger.error('error annotation_mention about ' + mention)
    # 返回字典
    return dic


def Annotate(txt, language="en", theta=0.1):
    """
    解决文本的概念实体与维基百科概念之间的映射问题
    :param txt: 一段文本对象，str类型
    :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
    :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
    :return:键值对[(A, B):score]  A为文本当中的概念实体，B为维基概念实体，score为其得分
    """
    '''
    一个使用 TagMe API 进行文本实体标注的函数调用。
    它将输入的文本 txt 中的实体与维基百科中的概念实体进行匹配，并返回一个字典，
    其中键值对表示匹配到的实体 对应的维基概念实体 它们的匹配得分。
    '''
    annotations = tagme.annotate(txt, lang=language)# 使用TagMe API进行文本实体标注
    dic = dict()
    try:
        for ann in annotations.get_annotations(theta):
            # print(ann)
            try:
                A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
                dic[(A, B)] = score
                # 将匹配到的实体对应的维基概念实体及其匹配得分添加到字典中
            except:
                logger.error('error annotation about ' + ann)
    except:
        pass
    return dic


if __name__ == '__main__':
    import spacy
    import time
    file = r"data-concept-instance-relations.txt"
    k = 5
    #如果 k 值越大，将会匹配更多维基概念实体，并可能导致返回更多的概念实体结果，从而会增加可能的噪声和不确定性。
    #如果 k 值越小，则可能会使匹配到的维基概念实体过于狭窄，但也能更准确地提供与文本相关的概念实体结果。
    nlp = spacy.load("en_core_web_sm")
    ent_concept = get_instance_concept(file)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
    f_w = open('train.tsv', 'w', encoding='utf-8')
    text = 'None'
    f = pd.read_csv('train.csv')
    for ii, line in enumerate(f['text']):

        label = f['labels'][ii]
        line = line.strip()
        text = line
        # text = ' '.join(line[:-1])
        # label = line[-1]
        # txt = 'Jay and Jolin are born in Taiwan'
        # obj = Annotate(text, theta=0.1)

        doc = nlp(text)
        obj = []

        for ent in doc.ents: # 遍历文本对象中所有实体
            obj.append(ent.text) #将 实体的文本内容添加到列表obj中
            # 输出实体的本文内容、起始字符位置、结束字符位置、实体类型
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        concept = []
        # 如果obj列表为空，则将字符串'None'添加到列表concept中
        if len(obj) == 0:
            concept.append('None')

        for ent in obj:
            # concept_sen = "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance="+i+"&topK=3"
            # response = requests.get(concept_sen, headers=headers)

            # content = json.loads(response.content)
            if ent in ent_concept:
                length = len(ent_concept[ent]) # 如果实体在字典中 则将该实体对应的概念实体添加到concept列表中
                length = k if length > k else length # length变量用于控制添加到concept列表中的概念实体的数量
                concept.extend(ent_concept[ent][0:length]) # k变量是一个阈值，用于控制匹配到的维基概念实体的数量
                
            else:
                concept.append(ent) # 否则将该实体本身添加到concept列表中
        f_w.write(str(text) + '\t' + ' '.join(concept) + '\t' + str(label) + '\n')


                # if len(content) > 0:
                #     # print(content.keys())
                #     concept.extend(content.keys())
                # else:
                #     concept.append(i)
                # time.sleep(0.5)
            # print(concept)
            # print(' '.join(concept))
            
    f_w.close()

