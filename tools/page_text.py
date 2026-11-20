"""Plain text of a rendered page (for proof-reading claims against numbers)."""
import html, re, sys
s = open(sys.argv[1]).read()
s = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
s = re.sub(r'<div class="cell-code.*?</div>', "", s, flags=re.S)
s = re.sub(r"<details.*?</details>", "", s, flags=re.S)
t = html.unescape(re.sub(r"<[^>]+>", "", s))
t = re.sub(r"\n\s*\n+", "\n\n", t)
i = t.find(sys.argv[2]) if len(sys.argv) > 2 else 0
print(t[i:i + int(sys.argv[3]) if len(sys.argv) > 3 else None])
