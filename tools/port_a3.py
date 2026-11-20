"""One-off: port the A3 report's R chunks to Python (see convert_inline.py)."""
import re
t = open("tools/work/a3.qmd").read()
t = t.replace('`{python} f"{g19[1]:.1f}"`', '`{python} f"{g19[0]:.1f}"`').replace('`{python} f"{g19[2]:.0f}"`', '`{python} f"{g19[1]:.0f}"`')
t = t.replace('''`{python} f"{min(sapply(slow['AGLJ.J'], function(x) x[[3]])):.0f}"`''', '''`{python} f"{min(g for _, _, g in slow['AGLJ.J']):.0f}"`''')
helpers = {
'cr <- function': '''def cr(ric, cls, col):
    return f"{100 * cls_row(ric, cls, 'volume share'):.1f}%"

def pct(x):
    return f"{100 * x:.0f}%"''',
'mt <- function': '''def mt(ric, f=np.median):
    return f"{100 * f(match[ric]):.0f}%"

def ce(ric):
    return f"{total_ex[ric]:,}"''',
'rr <- function': '''def rr(ric):
    return f"{rho[ric]:.2f}"

def rrd(ric, f):
    return f"{f(rho_days[ric]):.2f}"

def lrr(ric):
    return f"{round(lr[ric]):,}"''',
'ef <- function': '''def ef(ric, s, t, what="Gamma", d=2):
    return f"{eff(ric, s, t, what):.{d}f}"

def efms(ric, s, t):
    h = eff(ric, s, t, "half_life")
    return f"{1000 * h:.0f} ms" if h < 1 else f"{h:.1f} s"''',
'nks <- function': '''def nks(ric):
    k = ks_pass[ric]
    return "No" if k == 0 else str(k)

def mr(ric, i):
    return f"{mean_res[ric][i - 1]:.3f}"''',
}
L = t.split("\n"); out = []; i = 0
while i < len(L):
    l = L[i]
    if l.startswith("```{r"):
        j = i + 1
        while not L[j].startswith("```"): j += 1
        body = "\n".join(L[i+1:j])
        if "cat(py$LAST_TABLE)" in body:
            opts = [x for x in L[i+1:j] if x.startswith("#|")]
            out += ["```{python}"] + opts + ["print(LAST_TABLE)", "```"]
        else:
            key = next((k for k in helpers if k in body), None)
            if key:
                out += ["```{python}", "#| include: false", helpers[key], "```"]
        i = j + 1; continue
    out.append(l); i += 1
t = "\n".join(out)
assert "`r " not in t
open("tools/work/a3_py.qmd", "w").write(t)
for i, l in enumerate(t.split("\n")):
    if re.match(r"^#{1,2} [A-Z]", l): print(i, l)
