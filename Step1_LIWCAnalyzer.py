# -*- coding: utf-8 -*-
# @Time        : 2026.03.23
# @Author      : Hedy Hu
# @File        : Step1_LIWCAnalyzer.py
# @Software    : PyCharm, Python 3.11.14, lmsys Conda Env
# @Requirement : liwc == '0.5.0' pandas

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import liwc


PARENT_TO_CHILDREN = {
    "drives": ["affiliation", "achieve", "power"],
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def read_dic_word_categories(dic_path: str) -> dict[str, set[str]]:
    categories = {}
    word_to_cats = {}
    section = 0

    with open(dic_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "%":
                section += 1
                continue

            parts = line.split("\t")
            if section == 1:
                categories[parts[0]] = parts[1]
            elif section >= 2:
                word_to_cats[parts[0]] = {categories[cid] for cid in parts[1:] if cid in categories}

    return word_to_cats


def analyze_with_parser(text: str, parse_func) -> dict:
    tokens = tokenize(text)
    total_tokens = len(tokens)
    token_details = []
    category_counts = {}

    for token in tokens:
        cats = list(parse_func(token))
        token_details.append({"token": token, "categories": cats})
        for cat in set(cats):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    scores = {
        cat: count / total_tokens * 100 if total_tokens else 0.0
        for cat, count in category_counts.items()
    }

    return {
        "text": text,
        "tokens": tokens,
        "token_count": total_tokens,
        "token_details": token_details,
        "scores": scores,
    }


def token_level_union_score(result: dict, categories: list[str]) -> float:
    total_tokens = result["token_count"]
    if total_tokens == 0:
        return 0.0

    matched = 0
    for item in result["token_details"]:
        if any(cat in item["categories"] for cat in categories):
            matched += 1
    return matched / total_tokens * 100


def matched_tokens_for_categories(result: dict, categories: list[str]) -> list[dict]:
    rows = []
    for item in result["token_details"]:
        hits = [cat for cat in item["categories"] if cat in categories]
        if hits:
            rows.append({"token": item["token"], "hits": hits})
    return rows


def actual_dictionary_membership(word_to_cats: dict[str, set[str]], parent: str, children: list[str]) -> dict:
    parent_words = {word for word, cats in word_to_cats.items() if parent in cats}
    child_words = {
        child: {word for word, cats in word_to_cats.items() if child in cats}
        for child in children
    }
    union_children = set().union(*child_words.values()) if child_words else set()

    return {
        "parent_word_count": len(parent_words),
        "child_word_counts": {child: len(words) for child, words in child_words.items()},
        "children_sum_count": sum(len(words) for words in child_words.values()),
        "children_union_count": len(union_children),
        "parent_only_count": len(parent_words - union_children),
        "child_only_count": len(union_children - parent_words),
        "overlap_count": len(parent_words & union_children),
        "parent_only_examples": sorted(parent_words - union_children)[:30],
        "child_only_examples": sorted(union_children - parent_words)[:30],
        "overlap_examples": sorted(parent_words & union_children)[:30],
    }


def compare_parent_theory(text: str, parent: str, children: list[str], raw_dic: str, aug_dic: str) -> dict:
    raw_parse, _ = liwc.load_token_parser(raw_dic)
    aug_parse, _ = liwc.load_token_parser(aug_dic)

    raw_result = analyze_with_parser(text, raw_parse)
    aug_result = analyze_with_parser(text, aug_parse)

    raw_word_to_cats = read_dic_word_categories(raw_dic)
    membership = actual_dictionary_membership(raw_word_to_cats, parent, children)

    raw_parent_score = raw_result["scores"].get(parent, 0.0)
    raw_child_scores = {child: raw_result["scores"].get(child, 0.0) for child in children}
    raw_children_sum = sum(raw_child_scores.values())
    raw_children_union = token_level_union_score(raw_result, children)
    raw_parent_plus_children_union = token_level_union_score(raw_result, [parent] + children)
    augmented_parent_score = aug_result["scores"].get(parent, 0.0)

    return {
        "text": text,
        "parent": parent,
        "children": children,
        "dictionary_membership": membership,
        "raw_parent_score": raw_parent_score,
        "raw_child_scores": raw_child_scores,
        "raw_children_sum": raw_children_sum,
        "raw_children_union": raw_children_union,
        "raw_parent_plus_children_union": raw_parent_plus_children_union,
        "augmented_parent_score": augmented_parent_score,
        "raw_parent_tokens": matched_tokens_for_categories(raw_result, [parent]),
        "raw_children_tokens": matched_tokens_for_categories(raw_result, children),
        "raw_parent_plus_children_tokens": matched_tokens_for_categories(raw_result, [parent] + children),
        "augmented_parent_tokens": matched_tokens_for_categories(aug_result, [parent]),
    }


def print_report(report: dict):
    print("=" * 100)
    print("TEXT")
    print(report["text"])
    print("=" * 100)
    print(f'Parent: {report["parent"]}')
    print(f'Children: {report["children"]}')
    print()

    print("[Dictionary membership based on the actual OCR dic]")
    membership = report["dictionary_membership"]
    print(f"  parent_word_count      = {membership['parent_word_count']}")
    for child, count in membership["child_word_counts"].items():
        print(f"  child_word_count[{child:<11}] = {count}")
    print(f"  children_sum_count     = {membership['children_sum_count']}")
    print(f"  children_union_count   = {membership['children_union_count']}")
    print(f"  parent_only_count      = {membership['parent_only_count']}")
    print(f"  child_only_count       = {membership['child_only_count']}")
    print(f"  overlap_count          = {membership['overlap_count']}")
    print(f"  parent_only_examples   = {membership['parent_only_examples']}")
    print(f"  child_only_examples    = {membership['child_only_examples']}")
    print(f"  overlap_examples       = {membership['overlap_examples']}")
    print()

    print("[Sentence-level score comparison]")
    print(f"  raw_parent_score               = {report['raw_parent_score']:.2f}")
    for child, score in report["raw_child_scores"].items():
        print(f"  raw_child_score[{child:<11}] = {score:.2f}")
    print(f"  raw_children_sum              = {report['raw_children_sum']:.2f}")
    print(f"  raw_children_union            = {report['raw_children_union']:.2f}")
    print(f"  raw_parent_plus_children_union= {report['raw_parent_plus_children_union']:.2f}")
    print(f"  augmented_parent_score        = {report['augmented_parent_score']:.2f}")
    print()

    print("[Token matches in the raw OCR dic]")
    print(f"  parent tokens                 = {report['raw_parent_tokens']}")
    print(f"  children tokens               = {report['raw_children_tokens']}")
    print(f"  parent+children tokens        = {report['raw_parent_plus_children_tokens']}")
    print()

    print("[Token matches in the augmented dic]")
    print(f"  parent tokens                 = {report['augmented_parent_tokens']}")
    print()

    print("[Interpretation]")
    print("  1) raw_children_sum 检查“直接相加 children”会不会重复计数。")
    print("  2) raw_parent_plus_children_union 才是“parent 自己的词 + children 词”的真实去重并集。")
    print("  3) augmented_parent_score 理论上应与 raw_parent_plus_children_union 接近。")


if __name__ == "__main__":
    raw_dic = Path("mid/English_LIWC2022 Dictionary.dic")
    aug_dic = Path("mid/English_LIWC2022 Dictionary_augmented.dic")

    text = "I want to achieve success and gain power through teamwork."

    report = compare_parent_theory(
        text=text,
        parent="drives",
        children=PARENT_TO_CHILDREN["drives"],
        raw_dic=str(raw_dic),
        aug_dic=str(aug_dic),
    )
    print_report(report)

    summary_df = pd.DataFrame([
        {
            "text": report["text"],
            "raw_parent_score": report["raw_parent_score"],
            "raw_affiliation": report["raw_child_scores"].get("affiliation", 0.0),
            "raw_achieve": report["raw_child_scores"].get("achieve", 0.0),
            "raw_power": report["raw_child_scores"].get("power", 0.0),
            "raw_children_sum": report["raw_children_sum"],
            "raw_children_union": report["raw_children_union"],
            "raw_parent_plus_children_union": report["raw_parent_plus_children_union"],
            "augmented_parent_score": report["augmented_parent_score"],
            "parent_word_count": report["dictionary_membership"]["parent_word_count"],
            "children_union_count": report["dictionary_membership"]["children_union_count"],
            "parent_only_count": report["dictionary_membership"]["parent_only_count"],
            "child_only_count": report["dictionary_membership"]["child_only_count"],
            "overlap_count": report["dictionary_membership"]["overlap_count"],
        }
    ])
    summary_df.to_excel("mid/Step1_drives_theory_check.xlsx", index=False)
    print("已导出: mid/Step1_drives_theory_check.xlsx")
