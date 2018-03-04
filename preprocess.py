# coding: utf-8

import os
import re
import gensim
from hanziconv import HanziConv

try:
    import ujson as json
except:
    import json

glove = gensim.models.KeyedVectors.load_word2vec_format("model/vectors.txt.glove", binary=False)
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

def t2s():
    counter = 0
    dirs = os.listdir("data/json")
    for file in dirs:
        with open("data/json/" + file, "r", encoding='utf-8') as f:
            poems = json.load(f)
            for poem in poems:
                paragraphs = ""
                for sentence in poem.get("paragraphs"):
                    paragraphs += sentence
                poem["paragraphs"] = HanziConv.toSimplified(paragraphs)
                poem["title"] = HanziConv.toSimplified(poem.get("title"))
                title = poem["title"].split(" ")
                poem["title"] = punctuation_filter(title[0])
                del poem["strains"]
                del poem["author"]

        # 用以训练词向量
        with open("data/simplified_json/" + file, "w", encoding='utf-8') as fout_s:
            json.dump(poems, fout_s, ensure_ascii = False)

        filtered_poems = []
        for poem in poems:
            if len(poem["paragraphs"]) > 16 and len(poem["paragraphs"]) % 4 == 0 and "〖" not in poem["paragraphs"] and "（" not in poem["paragraphs"] and "[" not in poem["paragraphs"] and "｛" not in poem["paragraphs"] and  "{" not in poem["paragraphs"] and  "-" not in poem["paragraphs"] and "" not in poem["paragraphs"] and "《" not in poem["paragraphs"] and "「" not in poem["paragraphs"]  and "”" not in poem["paragraphs"]  and "" not in poem["paragraphs"] and "●" not in poem["paragraphs"]:
                sentences = re.split(r'[，。！？]', poem["paragraphs"])
                sentences = [sen + '，' for sen in sentences if len(sen) > 0]
                train_flag = True    
                if len(sentences) < 9:
                    for sen in sentences:
                            #print(sen)
                        if not train_flag:
                            break
                        if len(sen) != 6 and len(sen) !=8:
                            train_flag = False

                        for word in sen:
                            try:
                                vec = glove[word]
                            except:
                                train_flag = False
                                break
                if train_flag:
                    filtered_poems.append(poem)

        # 用以训练古诗生成模型
        with open("data/filtered_json/" + file, "w", encoding='utf-8') as fout_f:
            json.dump(filtered_poems, fout_f, ensure_ascii = False)

    print(counter)


def one_hot():
    counter = 0
    characters = {}
    dirs = os.listdir("data/filtered_json")
    for file in dirs:
        with open("data/filtered_json/" + file, "r", encoding='utf-8') as f:
            poems = json.load(f)
            for poem in poems:
                for c in poem["title"]:
                    if c not in characters:
                        characters[c] = counter
                        print(c, counter)
                        counter += 1

                for cp in poem["paragraphs"]:
                    if cp not in characters:
                        characters[cp] = counter
                        print(cp, counter)
                        counter += 1

    with open("data/characters.json", "w", encoding="utf-8") as fout:
        json.dump(characters, fout, ensure_ascii=False)


def corpus():
    dirs = os.listdir("data/simplified_json")
    #dirs = ["poet.song.0.json"]
    with open("data/corpus.txt", "w", encoding="utf-8") as fout:
        for file in dirs:
            with open("data/simplified_json/" + file, "r", encoding='utf-8') as f:
                poems = json.load(f)
                for poem in poems:
                    title = poem["title"].split(" ")
                    #print(poem["title"] + " --- " + title[0])
                    # words = punctuation_filter(title[0] + poem["paragraphs"])
                    words = title[0] + poem["paragraphs"]
                    for word in words:
                        fout.write(word + " ")
                    fout.write("\n")


def punctuation_filter(article):
    result = zhPattern.findall(article)
    title_filtered = ''
    if result:
        for words in result:
            title_filtered += words
        return title_filtered

    return article


if __name__ == "__main__":
    #t2s()
    #one_hot()
    corpus()