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
from collections import defaultdict
from pathlib import Path

import pandas as pd


PARENT_TO_CHILDREN = {
    "linguistic": [
        "function",
        "pronoun",
        "ppron",
        "i",
        "we",
        "you",
        "shehe",
        "they",
        "ipron",
        "det",
        "article",
        "number",
        "prep",
        "auxverb",
        "adverb",
        "conj",
        "negate",
        "verb",
        "adj",
        "quantity",
    ],
    "pronoun": ["ppron", "ipron"],
    "ppron": ["i", "we", "you", "shehe", "they"],
    "drives": ["affiliation", "achieve", "power"],
    "cognition": ["allnone", "cogproc", "insight", "cause", "discrep", "tentat", "certitude", "differ", "memory"],
    "affect": ["tone_pos", "tone_neg", "emotion", "emo_pos", "emo_neg", "emo_anx", "emo_anger", "emo_sad", "swear"],
    "social": ["socbehav", "prosocial", "polite", "conflict", "moral", "comm", "socrefs", "family", "friend", "female", "male"],
    "culture": ["politic", "ethnicity", "tech"],
    "lifestyle": ["leisure", "home", "work", "money", "relig"],
    "physical": [
        "health",
        "illness",
        "wellness",
        "mental",
        "substances",
        "sexual",
        "food",
        "death",
        "need",
        "want",
        "acquire",
        "lack",
        "fulfill",
        "fatigue",
        "reward",
        "risk",
        "curiosity",
        "allure",
    ],
    "perception": ["attention", "motion", "space", "visual", "auditory", "feeling", "time", "focuspast", "focuspresent", "focusfuture"],
    "conversation": ["netspeak", "assent", "nonflu", "filler"],
    "allpunc": ["period", "comma", "qmark", "exclam", "apostro", "otherp", "emoji"],
}


CATEGORY_LABELS = {
    "analytic": "Analytic Thinking",
    "clout": "Clout",
    "authentic": "Authenticity",
    "tone": "Emotional Tone",
    "linguistic": "Linguistic",
    "function": "Function Words",
    "pronoun": "Pronouns",
    "ppron": "Personal Pronouns",
    "i": "First Person Singular",
    "we": "First Person Plural",
    "you": "Second Person",
    "shehe": "Third Person Singular",
    "they": "Third Person Plural",
    "ipron": "Impersonal Pronouns",
    "det": "Determiners",
    "article": "Articles",
    "number": "Numbers",
    "prep": "Prepositions",
    "auxverb": "Auxiliary Verbs",
    "adverb": "Adverbs",
    "conj": "Conjunctions",
    "negate": "Negations",
    "verb": "Verbs",
    "adj": "Adjectives",
    "quantity": "Quantity",
    "drives": "Drives",
    "affiliation": "Affiliation",
    "achieve": "Achievement",
    "power": "Power",
    "cognition": "Cognition",
    "allnone": "All-or-none",
    "cogproc": "Cognitive Processes",
    "insight": "Insight",
    "cause": "Causation",
    "discrep": "Discrepancy",
    "tentat": "Tentative",
    "certitude": "Certainty",
    "differ": "Differentiation",
    "memory": "Memory",
    "affect": "Affect",
    "tone_pos": "Positive Tone",
    "tone_neg": "Negative Tone",
    "emotion": "Emotion",
    "emo_pos": "Positive Emotion",
    "emo_neg": "Negative Emotion",
    "emo_anx": "Anxiety",
    "emo_anger": "Anger",
    "emo_sad": "Sadness",
    "swear": "Swear",
    "social": "Social",
    "socbehav": "Social Behavior",
    "prosocial": "Prosocial",
    "polite": "Politeness",
    "conflict": "Conflict",
    "moral": "Moral",
    "comm": "Communication",
    "socrefs": "Social Referents",
    "family": "Family",
    "friend": "Friends",
    "female": "Female References",
    "male": "Male References",
    "culture": "Culture",
    "politic": "Politics",
    "ethnicity": "Ethnicity",
    "tech": "Technology",
    "lifestyle": "Lifestyle",
    "leisure": "Leisure",
    "home": "Home",
    "work": "Work",
    "money": "Money",
    "relig": "Religion",
    "physical": "Physical",
    "health": "Health",
    "illness": "Illness",
    "wellness": "Wellness",
    "mental": "Mental Health",
    "substances": "Substances",
    "sexual": "Sexual",
    "food": "Food",
    "death": "Death",
    "need": "Need",
    "want": "Want",
    "acquire": "Acquire",
    "lack": "Lack",
    "fulfill": "Fulfill",
    "fatigue": "Fatigue",
    "reward": "Reward",
    "risk": "Risk",
    "curiosity": "Curiosity",
    "allure": "Allure",
    "perception": "Perception",
    "attention": "Attention",
    "motion": "Motion",
    "space": "Space",
    "visual": "Visual",
    "auditory": "Auditory",
    "feeling": "Feeling",
    "time": "Time",
    "focuspast": "Past Focus",
    "focuspresent": "Present Focus",
    "focusfuture": "Future Focus",
    "conversation": "Conversation",
    "netspeak": "Netspeak",
    "assent": "Assent",
    "nonflu": "Nonfluencies",
    "filler": "Fillers",
    "allpunc": "All Punctuation",
    "period": "Period",
    "comma": "Comma",
    "qmark": "Question Mark",
    "exclam": "Exclamation Mark",
    "apostro": "Apostrophe",
    "otherp": "Other Punctuation",
    "emoji": "Emoji",
}


def load_word_category_pairs(csv_df: pd.DataFrame) -> dict[str, set[str]]:
    """从 csv_df 中读取 word 和 category，返回 {word: set(categories)}。"""
    word_to_cats = defaultdict(set)
    for _, row in csv_df.iterrows():
        word = row["word"].strip().lower()
        cat = row["category"].strip().lower()
        if word and cat:
            word_to_cats[word].add(cat)
    return word_to_cats


def is_supported_liwc_token(word: str) -> bool:
    clean_word = word.strip().lower()
    if not clean_word:
        return False
    if " " in clean_word or "\t" in clean_word:
        return False
    if clean_word.count("*") > 1:
        return False
    if "*" in clean_word and not clean_word.endswith("*"):
        return False
    return True


def write_liwc_dict(word_to_cats, cat_to_num, output_path):
    num_to_cat = {int(v): k for k, v in cat_to_num.items()}
    sorted_nums = sorted(num_to_cat.keys())

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%\n")
        for num in sorted_nums:
            f.write(f"{num}\t{num_to_cat[num]}\n")
        f.write("%\n")

        for word in sorted(word_to_cats.keys()):
            clean_word = word.strip().lower()
            if not is_supported_liwc_token(clean_word):
                continue

            cats = word_to_cats[word]
            codes = sorted([str(cat_to_num[c]) for c in cats if c in cat_to_num], key=int)

            if codes:
                words_string = "\t".join(codes)
                f.write(f"{clean_word}\t{words_string}\n")


def build_child_to_ancestors(parent_to_children: dict[str, list[str]]) -> dict[str, set[str]]:
    child_to_ancestors = defaultdict(set)

    for parent, children in parent_to_children.items():
        for child in children:
            child_to_ancestors[child].add(parent)

    changed = True
    while changed:
        changed = False
        for child, ancestors in list(child_to_ancestors.items()):
            expanded = set(ancestors)
            for anc in list(ancestors):
                expanded |= child_to_ancestors.get(anc, set())
            if expanded != ancestors:
                child_to_ancestors[child] = expanded
                changed = True

    return dict(child_to_ancestors)


def augment_word_to_cats(word_to_cats: dict[str, set[str]], parent_to_children: dict[str, list[str]]) -> dict[str, set[str]]:
    child_to_ancestors = build_child_to_ancestors(parent_to_children)
    augmented = {}
    for word, cats in word_to_cats.items():
        expanded = set(cats)
        for cat in list(cats):
            expanded |= child_to_ancestors.get(cat, set())
        augmented[word] = expanded
    return augmented


def build_child_to_parent(parent_to_children: dict[str, list[str]]) -> dict[str, str]:
    child_to_parent = {}
    for parent, children in parent_to_children.items():
        for child in children:
            child_to_parent[child] = parent
    return child_to_parent


def compute_depth(cat: str, child_to_parent: dict[str, str]) -> int:
    depth = 0
    cur = cat
    while cur in child_to_parent:
        cur = child_to_parent[cur]
        depth += 1
    return depth


def make_display_label(cat: str) -> str:
    if cat in CATEGORY_LABELS:
        return CATEGORY_LABELS[cat]
    return cat.replace("_", " ").title()


def write_visual_liwc_dict(word_to_cats, cat_to_num, output_path, parent_to_children):
    num_to_cat = {int(v): k for k, v in cat_to_num.items()}
    sorted_cats = [num_to_cat[num] for num in sorted(num_to_cat.keys())]
    child_to_parent = build_child_to_parent(parent_to_children)

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%\n")
        for cat in sorted_cats:
            indent = "    " * compute_depth(cat, child_to_parent)
            label = make_display_label(cat)
            f.write(f"{indent}{cat_to_num[cat]}\t{cat} ({label})\n")
        f.write("%\n")

        for word in sorted(word_to_cats.keys()):
            clean_word = word.strip().lower()
            if not is_supported_liwc_token(clean_word):
                continue

            cats = word_to_cats[word]
            codes = sorted([str(cat_to_num[c]) for c in cats if c in cat_to_num], key=int)

            if codes:
                f.write(f"{clean_word}\t" + "\t".join(codes) + "\n")


if __name__ == "__main__":
    input_csv1_original = pd.read_csv("mid/output_1st/liwc_words_final.csv")
    input_csv1_resume = pd.read_csv("mid/output_1st/liwc_words_final_resume.csv")
    input_csv1 = pd.concat([input_csv1_original, input_csv1_resume]).drop_duplicates(["word", "category"], keep="first")

    input_csv2 = pd.read_csv("mid/output_2nd/liwc_words_final.csv")
    common_pairs = pd.merge(
        input_csv1[["word", "category"]],
        input_csv2[["word", "category"]],
        on=["word", "category"],
        how="inner",
    ).drop_duplicates(["word", "category"])

    print(f"交集词条-类别对数量：{len(common_pairs)}")
    print(f"词典对应总共词数：{len(set(common_pairs['word']))}")

    valid_liwc_categories = {
        c.lower() for c in {
            "Analytic", "Clout", "Authentic", "Tone",
            "Linguistic", "function", "pronoun", "ppron",
            "i", "we", "you", "shehe", "they", "ipron",
            "det", "article", "number", "prep", "auxverb",
            "adverb", "conj", "negate", "verb", "adj", "quantity",
            "Drives", "affiliation", "achieve", "power",
            "Cognition", "allnone", "cogproc", "insight", "cause",
            "discrep", "tentat", "certitude", "differ", "memory",
            "Affect", "tone_pos", "tone_neg", "emotion",
            "emo_pos", "emo_neg", "emo_anx", "emo_anger", "emo_sad",
            "swear",
            "Social", "socbehav", "prosocial", "polite", "conflict",
            "moral", "comm", "socrefs", "family", "friend",
            "female", "male",
            "Culture", "politic", "ethnicity", "tech",
            "Lifestyle", "leisure", "home", "work", "money", "relig",
            "Physical", "health", "illness", "wellness", "mental",
            "substances", "sexual", "food", "death",
            "need", "want", "acquire", "lack", "fulfill",
            "fatigue", "reward", "risk", "curiosity", "allure",
            "Perception", "attention", "motion", "space",
            "visual", "auditory", "feeling",
            "time", "focuspast", "focuspresent", "focusfuture",
            "Conversation", "netspeak", "assent", "nonflu", "filler",
            "AllPunc", "Period", "Comma", "QMark", "Exclam",
            "Apostro", "OtherP", "Emoji",
        }
    }

    category_to_number = {}
    for cnt, item in enumerate(sorted(valid_liwc_categories), start=1):
        category_to_number[item] = cnt

    word_to_cats = load_word_category_pairs(common_pairs)
    augmented_word_to_cats = augment_word_to_cats(word_to_cats, PARENT_TO_CHILDREN)

    output_dic = Path("mid/English_LIWC2022 Dictionary.dic")
    write_liwc_dict(word_to_cats, category_to_number, output_dic)
    print(f"已生成 OCR词典: {output_dic}")

    output_aug_dic = Path("mid/English_LIWC2022 Dictionary_augmented.dic")
    write_liwc_dict(augmented_word_to_cats, category_to_number, output_aug_dic)
    print(f"已生成增强版词典: {output_aug_dic}")

    output_visual_dic = Path("mid/English_LIWC2022 Dictionary_visual.dic")
    write_visual_liwc_dict(word_to_cats, category_to_number, output_visual_dic, PARENT_TO_CHILDREN)
    print(f"已生成可视化词典: {output_visual_dic}")
