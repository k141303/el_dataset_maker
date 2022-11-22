import re
import os

import gzip
import json

import argparse

from multiprocessing import Pool
import multiprocessing as multi

import tqdm


def load_args():
    parser = argparse.ArgumentParser(description="CirrusDumpからEntityLinkingデータセットを作成")
    parser.add_argument("cirrus_path", help="CirrusDumpのパス")
    parser.add_argument("--output_dir", default="dataset")
    return parser.parse_args()


def remove_nested_brackets(sentence, start="\{\{", end="\}\}"):
    offsets = [(m.span(0)[0], 1) for m in re.finditer(start, sentence)]
    offsets += [(m.span(0)[1], -1) for m in re.finditer(end, sentence)]
    offsets = sorted(offsets, key=lambda x: x[0])

    nest = 0
    spans = []
    for cur, flag in offsets:
        if nest == 0:
            if flag < 0:
                continue
            spans.append([cur])
        nest += flag
        if nest == 0:
            spans[-1].append(cur)

    spans = [span for span in spans if len(span) == 2]

    for s, e in reversed(spans):
        sentence = sentence[:s] + sentence[e:]

    return sentence


def clean_line_level(sentence):
    clean_sentence = []
    for line in sentence.splitlines():
        line = line.strip()
        m = re.match("^:*?(\*|#)", line)
        if m is not None:  # 箇条書きは除去
            continue
        line = re.sub("'''", "", line)  # 強調記号の削除
        line = re.sub("''", "", line)  # 斜線記号の削除
        line = re.sub("([^\[]|^)\[[^\[\]].*?\]([^\]]|$)", "\\1\\2", line)  # 連続しない角括弧に囲まれた範囲を削除
        # line = re.sub("\[http.*?\]", "", line)  # 外部リンクの削除
        line = re.sub("\[\[Category.*?\]\]", "", line)  # カテゴリーの削除
        line = re.sub("\[\[(File|Image|画像):.*?\]\]", "", line)  # ファイルの削除
        line = line.replace("&nbsp;", "")  # NBSの削除

        line = re.sub("={2,10}.*?={2,10}", "", line)  # 見出しの削除

        line = line.strip()
        if len(line):
            clean_sentence.append(line)
    return "".join(clean_sentence)


def clean_source_text(source_text):
    source_text = remove_nested_brackets(source_text, start="\{\{", end="\}\}")  # スクリプトの削除
    source_text = remove_nested_brackets(source_text, start="\{\|", end="\|\}")  # スクリプトの削除
    source_text = re.sub("<[^>]*?/>", "", source_text)  # 独立したHTMLタグの削除
    source_text = remove_nested_brackets(
        source_text, start="<ref.*?>", end="</ref>"
    )  # refタグセットと内部の削除
    source_text = remove_nested_brackets(
        source_text, start="<script.*?>", end="</script>"
    )  # スクリプトの削除
    # source_text = remove_nested_brackets(source_text, start="<span.*?>", end="</span>")  # spanタグセットと内部の削除
    source_text = remove_nested_brackets(source_text, start="<table.*?>", end="</table>")  # テーブルの削除
    source_text = remove_nested_brackets(source_text, start="<div.*?>", end="</div>")  # テーブルの削除
    source_text = remove_nested_brackets(source_text, start="<score.*?>", end="</score>")  # テーブルの削除
    source_text = re.sub("<.*?>", "", source_text)  # HTMLタグの削除
    source_text = remove_nested_brackets(source_text, start="<!--", end="-->")  # コメントの削除

    source_text = clean_line_level(source_text)
    return source_text


def extract_links(source_text):
    links = []
    shift = 0
    for m in re.finditer("\[\[(.*?)\]\]", source_text):
        link_s, link_e = m.span(0)[0] - shift, m.span(0)[1] - shift
        if "|" in m.group(1):
            try:
                dst, surf = m.group(1).split("|")
            except:
                source_text = source_text[:link_s] + source_text[link_e:]
                shift += link_e - link_s
                continue
        else:
            dst, surf = m.group(1), m.group(1)

        if len(surf.strip()) == 0:
            source_text = source_text[:link_s] + source_text[link_e:]
            shift += link_e - link_s
            continue

        l_spaces = len(re.match("^\s*", surf).group(0))
        r_spaces = len(re.search("\s*?$", surf).group(0))

        span = [link_s + l_spaces, link_s + len(surf) - r_spaces]

        source_text = source_text[:link_s] + surf + source_text[link_e:]
        shift += link_e - link_s - len(surf)

        if "#" in m.group(1):  # 記事内の一部へのリンクは除去
            continue
        links.append({"span": span, "title": dst, "surf": surf})

    for d in links:
        surf = source_text[d["span"][0] : d["span"][1]]
        assert d["surf"].strip() == surf, f"リンクと表層が異なります。'{d['surf']}'!='{surf}'"

    return source_text, links


def parse(header, context):
    header = json.loads(header)
    context = json.loads(context)

    d = {}
    d["pageid"] = header["index"]["_id"]
    d["title"] = context["title"]
    d["redirect"] = context["redirect"]
    d["text"] = context["text"]

    source_text = context["source_text"]
    source_text = clean_source_text(source_text)
    d["source_text"], d["link"] = extract_links(source_text)

    return json.dumps(d, ensure_ascii=False)


def process(inputs):
    return parse(*inputs)


def load_file(file_path):
    with gzip.open(file_path, mode="rt") as f:
        for header in f:
            context = next(f)
            yield (header, context)


def save_jsonl(inputs):
    data, i, output_dir = inputs
    output_path = os.path.join(output_dir, f"{i}.json")
    with open(output_path, "w") as f:
        f.write("\n".join(data))


def main():
    args = load_args()

    data = []
    with Pool(multi.cpu_count() - 1) as p, tqdm.tqdm() as t:
        for d in p.imap(process, load_file(args.cirrus_path)):
            data.append(d)
            t.update()

    sub_dir, _ = os.path.splitext(os.path.basename(args.cirrus_path))
    output_dir = os.path.join(args.output_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    sep_data = [(data[i : i + 1000], i // 1000, output_dir) for i in range(0, len(data), 1000)]
    with Pool(multi.cpu_count() - 1) as p, tqdm.tqdm(desc="Saving", total=len(sep_data)) as t:
        for _ in p.imap(save_jsonl, sep_data):
            t.update()


if __name__ == "__main__":
    main()
