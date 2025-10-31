"""Dataset preparation helpers

This script converts a TSV of (pseudo, code) pairs into model-ready formats.
It supports:
 - TSV -> JSONL (one JSON object per line: {"pseudo":..., "code":...})
 - JSONL -> LM plain text with separators used by training (default tokens)

Usage examples:
  python prepare_dataset.py -i spoc-train-py.tsv --to-jsonl
  python prepare_dataset.py -i spoc-train-py.tsv --to-lm

The script is conservative: it skips rows with <min-length> tokens and can skip a header row.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable
import pandas as pd


def normalize(text: str) -> str:
    return text.strip()


def read_tsv_rows(tsv_path: Path, skip_header: bool = False) -> Iterable[tuple]:
    """Yield (col0, col1, ... ) tuples for each TSV row."""
    with tsv_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and skip_header:
                # basic header skip
                continue
            parts = line.rstrip('\n').split('\t')
            if not parts:
                continue
            yield tuple(parts)


def parse_blocks(path: Path) -> list:
    """Parse file into blocks separated by one or more empty lines.

    Returns list of blocks, where each block is a list of lines (strings).
    The caller can then pair blocks as (pseudo_block, code_block).
    """
    blocks = []
    cur = []
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if line.strip() == '':
                # boundary between blocks
                if cur:
                    blocks.append(cur)
                    cur = []
                else:
                    # consecutive blank lines: just continue
                    continue
            else:
                cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def tsv_to_jsonl(tsv_path: Path, jsonl_out: Path, min_length: int = 5, skip_header: bool = False, mode: str = 'blocks'):
    """Convert input file to JSONL. Two supported modes:
    - 'tsv': each non-empty row is split by TAB; first two columns are pseudo and code
    - 'blocks': file is split into blocks by blank lines; consecutive block pairs are (pseudo, code)
    """
    out_count = 0
    with jsonl_out.open('w', encoding='utf-8') as fo:
        if mode == 'tsv':
            # Attempt to read header and stream-group rows by group keys if present.
            with tsv_path.open('r', encoding='utf-8') as f:
                # read header
                header_line = f.readline()
                if not header_line:
                    return 0
                headers = [h.strip() for h in header_line.rstrip('\n').split('\t')]
                # determine indices
                def idx(name):
                    try:
                        return headers.index(name)
                    except ValueError:
                        return None

                text_i = idx('text')
                code_i = idx('code')
                prob_i = idx('probid')
                sub_i = idx('subid')
                worker_i = idx('workerid')

                # if both text and code columns exist and grouping keys exist, group by them
                if text_i is not None and code_i is not None and (prob_i is not None and sub_i is not None):
                    cur_key = None
                    pseudo_lines = []
                    code_lines = []
                    for raw in f:
                        parts = raw.rstrip('\n').split('\t')
                        # pad parts to headers length
                        if len(parts) < len(headers):
                            parts += [''] * (len(headers) - len(parts))
                        key = (parts[prob_i].strip(), parts[sub_i].strip())
                        if cur_key is None:
                            cur_key = key
                        if key != cur_key:
                            # flush previous group
                            pseudo = normalize('\n'.join(pseudo_lines))
                            code = normalize('\n'.join(code_lines))
                            if len(pseudo) >= min_length and len(code) >= min_length:
                                obj = {"pseudo": pseudo, "code": code}
                                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                out_count += 1
                            # reset accumulators
                            pseudo_lines = []
                            code_lines = []
                            cur_key = key
                        # append current row pieces if present
                        txt = parts[text_i].strip() if text_i is not None and text_i < len(parts) else ''
                        cde = parts[code_i].strip() if code_i is not None and code_i < len(parts) else ''
                        if txt:
                            pseudo_lines.append(txt)
                        if cde:
                            code_lines.append(cde)

                    # flush last group
                    if cur_key is not None:
                        pseudo = normalize('\n'.join(pseudo_lines))
                        code = normalize('\n'.join(code_lines))
                        if len(pseudo) >= min_length and len(code) >= min_length:
                            obj = {"pseudo": pseudo, "code": code}
                            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            out_count += 1
                else:
                    # fallback: each row is a pair in first two columns
                    for raw in f:
                        parts = raw.rstrip('\n').split('\t')
                        if len(parts) < 2:
                            continue
                        pseudo = normalize(parts[0])
                        code = normalize(parts[1])
                        if len(pseudo) < min_length or len(code) < min_length:
                            continue
                        obj = {"pseudo": pseudo, "code": code}
                        fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        out_count += 1
        elif mode == 'blocks':
            blocks = parse_blocks(tsv_path)
            if len(blocks) < 2:
                return 0
            # pair consecutive blocks: (0,1), (2,3), ...
            pairs = len(blocks) // 2
            for i in range(pairs):
                pseudo_lines = blocks[2 * i]
                code_lines = blocks[2 * i + 1]
                pseudo = normalize('\n'.join(pseudo_lines))
                code = normalize('\n'.join(code_lines))
                if len(pseudo) < min_length or len(code) < min_length:
                    continue
                obj = {"pseudo": pseudo, "code": code}
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                out_count += 1
            # if there is an odd trailing block, ignore it but warn (print)
            if len(blocks) % 2 == 1:
                print(f"Warning: odd number of blocks ({len(blocks)}). The last block was ignored.")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return out_count


def jsonl_to_lm(jsonl_in: Path, lm_out: Path, pseudo_token: str = "<|pseudocode|>", code_token: str = "<|code|>"):
    PRE = f"{pseudo_token}\n"
    SEP = f"\n{code_token}\n"
    out_count = 0
    with jsonl_in.open('r', encoding='utf-8') as fi, lm_out.open('w', encoding='utf-8') as fo:
        for line in fi:
            if not line.strip():
                continue
            obj = json.loads(line)
            pseudo = obj.get('pseudo', '')
            code = obj.get('code', '')
            fo.write(PRE + pseudo + SEP + code + "\n\n")
            out_count += 1
    return out_count


def _collapse_text(s: str, mode: str) -> str:
    if s is None:
        return ''
    if mode == 'space':
        # replace any whitespace including newlines and tabs with a single space
        return ' '.join(s.split())
    else:
        # represent newlines explicitly with '\n' sequences, while collapsing other whitespace
        parts = [p.strip() for p in s.splitlines() if p is not None]
        return '\\n'.join([' '.join(p.split()) for p in parts])


def write_plain_from_jsonl(jsonl_in: Path, txt_out: Path, collapse_newlines: str = 'space', sep: str = '\t') -> int:
    """Write a plain text file where each example is one line: <pseudo><TAB><code>

    Internal newlines are collapsed according to collapse_newlines ('space' or '\\n').
    Tabs inside data are replaced with spaces.
    """
    out_count = 0
    with jsonl_in.open('r', encoding='utf-8') as fi, txt_out.open('w', encoding='utf-8') as fo:
        for line in fi:
            if not line.strip():
                continue
            obj = json.loads(line)
            pseudo = obj.get('pseudo', '')
            code = obj.get('code', '')
            pseudo = _collapse_text(pseudo, collapse_newlines).replace('\t', ' ')
            code = _collapse_text(code, collapse_newlines).replace('\t', ' ')
            fo.write(pseudo + sep + code + '\n')
            out_count += 1
    return out_count


def write_tagged_from_jsonl(jsonl_in: Path, txt_out: Path, pseudo_token: str = "<|pseudocode|>", code_token: str = "<|code|>") -> int:
    """Write a plain text file with tag-separated blocks (no JSON). Each example is written as:
    <|pseudocode|>\n{pseudo}\n<|code|>\n{code}\n\n
    This is the same tokens the LM expects but stored in plain text.
    """
    out_count = 0
    with jsonl_in.open('r', encoding='utf-8') as fi, txt_out.open('w', encoding='utf-8') as fo:
        for line in fi:
            if not line.strip():
                continue
            obj = json.loads(line)
            pseudo = obj.get('pseudo', '')
            code = obj.get('code', '')
            fo.write(pseudo_token + '\n' + pseudo + '\n' + code_token + '\n' + code + '\n\n')
            out_count += 1
    return out_count


def xlsx_to_jsonl(xlsx_path: Path, jsonl_out: Path, min_length: int = 5, sheet_name=0):
    """Read an Excel file and group rows into examples separated by fully-empty rows.

    Assumptions about the sheet layout:
    - Pseudo (left) is in the first column; code (right) is in the second column.
    - Examples are separated by one or more fully-empty rows (both columns empty).
    - Within an example, multiple rows in the left column are joined with '\n' to form the full pseudo,
      and multiple rows in the right column are joined with '\n' to form the full code.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=str)
    out_count = 0
    with jsonl_out.open('w', encoding='utf-8') as fo:
        pseudo_lines = []
        code_lines = []
        for _, row in df.iterrows():
            left = '' if pd.isna(row.get(0)) else str(row.get(0)).rstrip('\n')
            right = '' if pd.isna(row.get(1)) else str(row.get(1)).rstrip('\n')
            if (not left.strip()) and (not right.strip()):
                # separator: flush current block if any
                if pseudo_lines or code_lines:
                    pseudo = normalize('\n'.join(pseudo_lines))
                    code = normalize('\n'.join(code_lines))
                    if len(pseudo) >= min_length and len(code) >= min_length:
                        obj = {"pseudo": pseudo, "code": code}
                        fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        out_count += 1
                    pseudo_lines = []
                    code_lines = []
                # else just skip repeated empty rows
            else:
                if left.strip():
                    pseudo_lines.append(left)
                if right.strip():
                    code_lines.append(right)

        # flush last block
        if pseudo_lines or code_lines:
            pseudo = normalize('\n'.join(pseudo_lines))
            code = normalize('\n'.join(code_lines))
            if len(pseudo) >= min_length and len(code) >= min_length:
                obj = {"pseudo": pseudo, "code": code}
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                out_count += 1
    return out_count


def main():
    p = argparse.ArgumentParser(description="Convert TSV -> JSONL and/or LM-ready text for model training")
    p.add_argument('-i', '--input', required=True, help='Input TSV file (path)')
    p.add_argument('--to-jsonl', action='store_true', help='Produce a JSONL file from the TSV')
    p.add_argument('--to-lm', action='store_true', help='Produce LM-ready plain text output')
    p.add_argument('--to-txt', action='store_true', help='Produce a simple plain text file (one example per line, pseudo and code separated by a tab)')
    p.add_argument('--to-tagged', action='store_true', help='Produce a tagged plain text file using the same tags as LM output but without JSON (one block per example)')
    p.add_argument('--collapse-newlines', choices=['space', '\\n'], default='space', help="How to collapse internal newlines when writing plain text: 'space' (default) or '\\n' to keep explicit \n markers")
    p.add_argument('--mode', choices=['tsv', 'blocks'], default='blocks', help="Input file parsing mode: 'tsv' for per-row TSV, 'blocks' for blank-line separated blocks (default: blocks)")
    p.add_argument('--out-jsonl', help='Output JSONL path (default: input.jsonl)')
    p.add_argument('--out-lm', help='Output LM text path (default: input.lm.txt)')
    p.add_argument('--min-length', type=int, default=5, help='Minimum length for pseudo and code strings (default: 5)')
    p.add_argument('--skip-header', action='store_true', help='Skip first row (header) when reading TSV')
    args = p.parse_args()

    tsv_path = Path(args.input)
    if not tsv_path.exists():
        p.error(f"Input file not found: {tsv_path}")

    if not args.to_jsonl and not args.to_lm:
        # default: produce both
        args.to_jsonl = True
        args.to_lm = True

    jsonl_path = Path(args.out_jsonl) if args.out_jsonl else tsv_path.with_suffix('.jsonl')
    lm_path = Path(args.out_lm) if args.out_lm else tsv_path.with_suffix('.lm.txt')

    if args.to_jsonl:
        # select conversion function based on input extension and mode
        if tsv_path.suffix.lower() in ('.xlsx', '.xls'):
            cnt = xlsx_to_jsonl(tsv_path, jsonl_path, min_length=args.min_length)
        else:
            cnt = tsv_to_jsonl(tsv_path, jsonl_path, min_length=args.min_length, skip_header=args.skip_header, mode=args.mode)
        print(f"Wrote {cnt} examples to {jsonl_path}")

    if args.to_lm:
        # ensure we have a jsonl to convert from
        src_jsonl = jsonl_path if args.to_jsonl else tsv_path.with_suffix('.jsonl')
        if not src_jsonl.exists():
            # attempt to create jsonl on the fly
            if tsv_path.suffix.lower() in ('.xlsx', '.xls'):
                created = xlsx_to_jsonl(tsv_path, src_jsonl, min_length=args.min_length)
            else:
                created = tsv_to_jsonl(tsv_path, src_jsonl, min_length=args.min_length, skip_header=args.skip_header, mode=args.mode)
            print(f"Created {created} intermediate jsonl at {src_jsonl}")
        cnt2 = jsonl_to_lm(src_jsonl, lm_path)
        print(f"Wrote {cnt2} LM examples to {lm_path}")

    if args.to_txt:
        txt_path = Path(args.out_lm) if args.out_lm else tsv_path.with_suffix('.txt')
        # write plain tab-separated text (one example per line)
        collapsed = args.collapse_newlines
        cnt3 = write_plain_from_jsonl(src_jsonl, txt_path, collapse_newlines=collapsed)
        print(f"Wrote {cnt3} plain text examples to {txt_path}")

    if args.to_tagged:
        tagged_path = Path(args.out_lm) if args.out_lm else tsv_path.with_suffix('.tagged.txt')
        cnt4 = write_tagged_from_jsonl(src_jsonl, tagged_path)
        print(f"Wrote {cnt4} tagged text examples to {tagged_path}")


if __name__ == '__main__':
    main()
