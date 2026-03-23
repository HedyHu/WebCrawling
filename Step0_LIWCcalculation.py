# -*- coding: utf-8 -*-
# @Time        : 2026.03.21
# @Author      : Hedy Hu
# @File        : Step0_LIWCcalculation.py
# @Software    : PyCharm, Python 3.11.14, lmsys Conda Env
# @Requirement : pyliwc == '1.4' fastcore == '1.12.30' liwc == '0.5.0'
# @Reference   :
# Please remember that LIWC is an art of word count.
# 1.https://github.com/camille1/pyliwc
# 2.https://github.com/chbrown/liwc-python
# LIWC官方样例 https://www.liwc.app/help/cli
# 3.https://github.com/ryanboyd/liwc-22-cli-python/blob/main/LIWC-22-cli_Example.py
# 4.https://github.com/ryanboyd/liwc-22-cli-r/blob/main/LIWC-22-cli_Example.R
# 5.LIWC investigation https://claude.ai/chat/ad6b267e-3a77-42f2-aaa8-9e0db7bed06f
import pandas as pd
import csv
from collections import defaultdict
from pathlib import Path

def load_word_category_pairs(csv_df: pd.DataFrame) -> dict[str, set[str]]:
    """从 csv_df 中读取 word 和 category，返回 {word: set of categories}。"""
    word_to_cats = defaultdict(set)
    for _, row in csv_df.iterrows():
        word = row["word"].strip()
        cat = row["category"].strip().lower()
        if word and cat:
            word_to_cats[word].add(cat)
    return word_to_cats

from pathlib import Path
def write_liwc_dict(word_to_cats, cat_to_num, output_path):
    num_to_cat = {int(v): k for k, v in cat_to_num.items()}
    sorted_nums = sorted(num_to_cat.keys())

    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("%\n")
        for num in sorted_nums:
            f.write(f"{num}\t{num_to_cat[num]}\n")
        f.write("%\n")

        for word in sorted(word_to_cats.keys()):
            # 【关键改动】：只处理不含空格的词条
            # 这样 'n't your fault' 会被跳过，而 'abandon*' 会被保留
            if ' ' in word.strip():
                continue

            if '\t' in word.strip():
                continue

            clean_word = word.strip().lower()
            cats = word_to_cats[word]
            codes = sorted([str(cat_to_num[c]) for c in cats if c in cat_to_num])

            if codes:
                words_string = '\t'.join(codes)
                f.write(f"{clean_word}\t{words_string}\n")

if __name__ == "__main__":
    # Step1: Build liwc dict
    # output_1st and resume: strip-height 1000, upscale 2
    input_csv1_original = pd.read_csv("mid/output_1st/liwc_words_final.csv")
    input_csv1_resume = pd.read_csv("mid/output_1st/liwc_words_final_resume.csv")
    input_csv1 = pd.concat([input_csv1_original, input_csv1_resume]).drop_duplicates(["word","category"], keep="first")
    # output_2nd: strip-height 700, upscale 3, strip-overlap 100
    input_csv2 = pd.read_csv("mid/output_2nd/liwc_words_final.csv")
    # 取交集：只保留同时出现在两个数据集中的 (word, category)
    common_pairs = pd.merge(
        input_csv1[['word', 'category']],
        input_csv2[['word', 'category']],
        on=['word', 'category'],
        how='inner'
    ).drop_duplicates(['word', 'category'])

    print(f"交集词条-类别对数量：{len(common_pairs)}")
    # 交集词条-类别对数量：46693
    print(f"词典对应总共词数：{len(set(common_pairs['word']))}")
    # 词典对应总共词数：14192

    VALID_LIWC_CATEGORIES = {
    c.lower() for c in {
        "Analytic","Clout","Authentic","Tone",
        "Linguistic","function","pronoun","ppron",
        "i","we","you","shehe","they","ipron",
        "det","article","number","prep","auxverb",
        "adverb","conj","negate","verb","adj","quantity",
        "Drives","affiliation","achieve","power",
        "Cognition","allnone","cogproc","insight","cause",
        "discrep","tentat","certitude","differ","memory",
        "Affect","tone_pos","tone_neg","emotion",
        "emo_pos","emo_neg","emo_anx","emo_anger","emo_sad",
        "swear",
        "Social","socbehav","prosocial","polite","conflict",
        "moral","comm","socrefs","family","friend",
        "female","male",
        "Culture","politic","ethnicity","tech",
        "Lifestyle","leisure","home","work","money","relig",
        "Physical","health","illness","wellness","mental",
        "substances","sexual","food","death",
        "need","want","acquire","lack","fulfill",
        "fatigue","reward","risk","curiosity","allure",
        "Perception","attention","motion","space",
        "visual","auditory","feeling",
        "time","focuspast","focuspresent","focusfuture",
        "Conversation","netspeak","assent","nonflu","filler",
        "AllPunc","Period","Comma","QMark","Exclam",
        "Apostro","OtherP","Emoji",
    }}

    CATEGORY_TO_NUMBER = {}
    cnt = 0
    for item in sorted(list(VALID_LIWC_CATEGORIES)):  # 关键修复：加入 sorted
        if item not in CATEGORY_TO_NUMBER:
            CATEGORY_TO_NUMBER[item] = cnt + 1
            cnt += 1

    word_to_cats = load_word_category_pairs(common_pairs)

    # 调用
    output_dic = Path("mid/English_LIWC2022 Dictionary.dic")
    write_liwc_dict(word_to_cats, CATEGORY_TO_NUMBER, output_dic)
    print(f"已生成 {output_dic}") # 已生成 mid/English_LIWC2022 Dictionary.dic

    # Step2: Apply liwc analyzer
    import liwc
    parse, category_names = liwc.load_token_parser('mid/English_LIWC2022 Dictionary.dic')
    print("加载成功！类别数：", len(category_names)) # 加载成功！类别数： 114
    # 测试一个词条
    print(common_pairs[common_pairs['word'].str.contains("happy", na=False)])
    print(list(parse("happy")))  # 应返回类别名列表
