"""Development helper: execute a chapter's Python chunks in one namespace and
evaluate every inline `{python} expr`, reporting the ones that fail.

    JSEMICRO_DATA=synthetic python tools/check_inline.py chapters/06-cost-of-trading.qmd
"""
import os
import re
import sys
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")


def resolve(path):
    text = open(path).read()
    base = os.path.dirname(path)
    return re.sub(r"\{\{< include (\S+) >\}\}", lambda m: resolve(os.path.join(base, m.group(1))), text)


def main(path, run_chunks=True):
    text = resolve(path)
    os.chdir(os.path.dirname(os.path.abspath(path)))
    ns = {}
    if run_chunks:
        for i, m in enumerate(re.finditer(r"```\{python\}\n(.*?)```", text, re.S)):
            code = "\n".join(l for l in m.group(1).split("\n") if not l.startswith("#|"))
            try:
                exec(compile(code, f"chunk{i}", "exec"), ns)
            except Exception:
                print(f"--- chunk {i} failed:\n{code[:300]}")
                traceback.print_exc(limit=3)
                return
            import matplotlib.pyplot as plt
            plt.close("all")
    bad = 0
    for m in re.finditer(r"`\{python\} ([^`]+)`", text):
        try:
            eval(m.group(1), ns)
        except Exception as e:
            bad += 1
            print(f"FAIL {m.group(1)!r}: {type(e).__name__}: {e}")
    print(f"{bad} failing inline expressions")


if __name__ == "__main__":
    main(sys.argv[1])
