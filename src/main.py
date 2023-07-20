import re
import os
import bisect

import gzip
import json

import argparse

from collections import Counter

from multiprocessing import Pool
import multiprocessing as multi

import tqdm

from liat_ml_roberta import RoBERTaTokenizer


def load_args():
    parser = argparse.ArgumentParser(description="CirrusDumpからEntityLinkingデータセットを作成")
    parser.add_argument("cirrus_path", help="CirrusDumpのパス")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--model_name", default="roberta_base_ja_20190121_m10000_v24000_u125000")
    parser.add_argument("--debug_mode", action="store_true")
    return parser.parse_args()


def remove_nested_brackets(sentence, start="\{\{", end="\}\}", level=1):
    offsets = [(*m.span(0), 1) for m in re.finditer(start, sentence)]
    offsets += [(*m.span(0), -1) for m in re.finditer(end, sentence)]
    offsets = sorted(offsets, key=lambda x: (x[0]))

    nest = 0
    max_level = 0
    spans = []
    for s, e, flag in offsets:
        if nest == 0:
            if flag < 0:
                continue
            spans.append([s])
        nest += flag
        if nest > max_level:
            max_level = nest
        if nest == 0:
            spans[-1] += [e, max_level]
            max_level = 0

    spans = [span for span in spans if len(span) == 3]

    for s, e, max_level in reversed(spans):
        if max_level < level:
            continue
        sentence = sentence[:s] + sentence[e:]

    return sentence


def clean_line_level(sentence):
    clean_sentence = []
    for line in sentence.splitlines():
        line = line.strip()
        line = re.sub("^:*?(\*|#)", " ・ ", line)  # 箇条書きの置換
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


def replace_first_emphasis_to_self_link(source_text, title):
    m = re.search("('''(.*?)''')", source_text)
    if m is not None:
        s, e = m.span(1)
        source_text = source_text[:s] + f"[[{title}|{m.group(2)}]]" + source_text[e:]
    return source_text


def replace_first_yomigana_to_self_link(source_text, title):
    m = re.search("'''.*?'''\s*[』」]?\s*[\(（](.*?)[\)）]", source_text)
    if m is None:
        return source_text

    s, e = m.span(1)
    text = ""
    for yomi in re.split("([、:：])", m.group(1)):
        yomi = yomi.strip()
        m = re.match("^[\-.,'a-zA-Z\d\u3041-\u309F\u30A0-\u30FF\s]+$", yomi)
        if m is None:
            text += yomi
            continue
        text += f"[[{title}|{yomi}]]"

    source_text = source_text[:s] + text + source_text[e:]

    return source_text


def clean_source_text(source_text, title):
    source_text = remove_nested_brackets(source_text, start="\{\{", end="\}\}")  # スクリプトの削除
    source_text = remove_nested_brackets(source_text, start="\{\|", end="\|\}")  # スクリプトの削除
    source_text = remove_nested_brackets(source_text, start="\[\[", end="\]\]", level=2)  # ネストされたリンクの削除
    source_text = re.sub("<[^>]*?/>", "", source_text)  # 独立したHTMLタグの削除
    source_text = remove_nested_brackets(source_text, start="<ref.*?>", end="</ref>")  # refタグセットと内部の削除
    source_text = remove_nested_brackets(source_text, start="<script.*?>", end="</script>")  # スクリプトの削除
    # source_text = remove_nested_brackets(source_text, start="<span.*?>", end="</span>")  # spanタグセットと内部の削除
    # source_text = remove_nested_brackets(source_text, start="<table.*?>", end="</table>")  # テーブルの削除
    source_text = re.sub("</?(tr|td|th|div).*?>", " ", source_text)  # 表の罫線等を空白に変換
    # source_text = remove_nested_brackets(source_text, start="<div.*?>", end="</div>")  # テーブルの削除
    source_text = remove_nested_brackets(source_text, start="<score.*?>", end="</score>")  # テーブルの削除
    source_text = re.sub("<.*?>", "", source_text)  # HTMLタグの削除
    source_text = remove_nested_brackets(source_text, start="<!--", end="-->")  # コメントの削除

    source_text = replace_first_yomigana_to_self_link(source_text, title)
    source_text = replace_first_emphasis_to_self_link(source_text, title)
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
    if not str(d["pageid"]).isdigit():
        return None
    d["title"] = context["title"]
    d["redirect"] = context["redirect"]
    d["text"] = context["text"]

    source_text = context["source_text"]
    source_text = clean_source_text(source_text, d["title"])
    d["source_text"], d["link"] = extract_links(source_text)

    return d


def process(inputs):
    return parse(*inputs)


def load_file(file_path):
    with gzip.open(file_path, mode="rt") as f:
        for header in f:
            context = next(f)
            yield (header, context)


def save_jsonl(output_path, data):
    with open(output_path, "w") as f:
        json_dumps = lambda x: json.dumps(x, ensure_ascii=False)
        f.write("\n".join(map(json_dumps, data)))


def tokenize(inputs):
    data, data_id, output_dir, model_name = inputs

    tokenizer = RoBERTaTokenizer.from_pretrained(model_name)

    checks = []
    for d in data:
        # トークナイズの際に空白が削除されてしまうためオフセットを合わせる。
        space_offsets = [0]
        for m in re.finditer("\s", d["source_text"]):
            space_offsets.append(m.span(0)[1])

        tmp = re.sub("\s", "", d["source_text"])
        for i in range(len(d["link"])):
            link = d["link"][i]
            link["span"][0] -= bisect.bisect_right(space_offsets, link["span"][0]) - 1
            link["span"][1] -= bisect.bisect_right(space_offsets, link["span"][1]) - 1

            surf = re.sub("\s", "", link["surf"])
            span = tmp[link["span"][0] : link["span"][1]]
            assert surf == span, f"{surf} != {span}"

        # トークナイズ
        text = re.sub("@@", "@2", d["text"])
        source_text = re.sub("@@", "@2", d["source_text"])
        d["entity_tokens"] = tokenizer.tokenize(text)[:512]
        d["mention_tokens"] = tokenizer.tokenize(source_text)

        # リンクスパンをトークンスパンにマップ
        rm_head = lambda token: re.sub("@@$", "", token)
        orig_tokens = list(map(rm_head, d["mention_tokens"]))

        d["offset"] = [0]
        for token in orig_tokens:
            d["offset"].append(d["offset"][-1] + len(token))

        for link in d["link"]:
            s_off = bisect.bisect_left(d["offset"], link["span"][0])
            if d["offset"][s_off] != link["span"][0]:
                s_off -= 1
            e_off = bisect.bisect_left(d["offset"], link["span"][1])

            link["token_span"] = [s_off, e_off]

            try:
                check = d["offset"][s_off] != link["span"][0] or d["offset"][e_off] != link["span"][1]
            except IndexError:
                assert False, (d["title"], d["text"][-50:], link)
            checks.append(check)

        d["entity_token_ids"] = tokenizer.convert_tokens_to_ids(d["entity_tokens"])
        d["mention_token_ids"] = tokenizer.convert_tokens_to_ids(d["mention_tokens"])

        del d["text"]
        del d["source_text"]
        del d["offset"]
        del d["entity_tokens"]
        del d["mention_tokens"]

    output_path = os.path.join(output_dir, f"{data_id}.json")
    save_jsonl(output_path, data)

    return checks


def main():
    args = load_args()

    data = []
    with Pool(multi.cpu_count() - 1) as p, tqdm.tqdm() as t:
        for d in p.imap(process, load_file(args.cirrus_path), chunksize=500):
            if d is None:
                continue
            data.append(d)
            if args.debug_mode and len(data) >= 100000:
                break
            t.update()

    title2pageid = {}
    for d in data:
        title2pageid[d["title"]] = d["pageid"]
        for red in d["redirect"]:
            title2pageid[red["title"]] = d["pageid"]

    link_count = Counter()
    for d in data:
        for link in d["link"]:
            link["pageid"] = title2pageid.get(link["title"])
            if link["pageid"] is not None:
                link_count[link["pageid"]] += 1

    for d in data:
        d["count"] = link_count.get(d["pageid"], 0)

    sub_dir, _ = os.path.splitext(os.path.basename(args.cirrus_path))
    output_dir = os.path.join(args.output_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    n = 2000
    tasks = [(data[i : i + n], i // n, output_dir, args.model_name) for i in range(0, len(data), n)]

    checks = []
    with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Tokenizing", total=len(tasks)) as t:
        for _checks in p.imap_unordered(tokenize, tasks):
            t.update()
            checks += _checks

    offset_mismatch_rate = sum(checks) / len(checks) * 100
    print(f"Offset mismatch rate: {offset_mismatch_rate:.2f}%")


if __name__ == "__main__":
    main()
