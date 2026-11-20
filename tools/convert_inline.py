"""One-off helper used to port the knitr/reticulate reports to the Jupyter
engine: translates inline R expressions (`r py$x`) into inline Python
(`{python} x`). Kept for transparency; not part of the pipeline."""
import re
import sys


def py_ref(s):
    # py$a$b[["c"]] -> a["b"]["c"]
    def rep(m):
        name, rest = m.group(1), m.group(2) or ""
        rest = re.sub(r'\$([A-Za-z_][A-Za-z_0-9]*)', r"['\1']", rest)
        rest = re.sub(r'\[\[("[^"]*"|\d+)\]\]', lambda k: "[" + k.group(1).replace('"', "'") + "]", rest)
        return name + rest
    return re.sub(r'py\$([A-Za-z_][A-Za-z_0-9]*)((?:\$[A-Za-z_][A-Za-z_0-9]*|\[\[(?:"[^"]*"|\d+)\]\])*)', rep, s)


def convert_expr(e):
    e = e.strip()
    m = re.fullmatch(r'format\((.*), big\.mark = ","\)', e)
    if m:
        inner = convert_expr(m.group(1))
        return f'f"{{{inner}:,}}"'
    m = re.fullmatch(r'sprintf\("%\.(\d)f%%", 100 \* (.*)\)', e)
    if m:
        return f'f"{{100 * {convert_expr(m.group(2))}:.{m.group(1)}f}}%"'
    m = re.fullmatch(r'sprintf\("%\.(\d)f", (.*)\)', e)
    if m:
        return f'f"{{{convert_expr(m.group(2))}:.{m.group(1)}f}}"'
    return py_ref(e)


def convert(text):
    left = []

    def rep(m):
        out = convert_expr(m.group(1))
        if "py$" in out or "<-" in out or "sapply" in out or re.search(r'\[\d\]', out):
            left.append(m.group(0))
        return "`{python} " + out + "`"
    text = re.sub(r'`r ([^`]+)`', rep, text)
    return text, left


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    t, left = convert(open(src).read())
    open(dst, "w").write(t)
    print("needs manual attention:", *left, sep="\n  ")
