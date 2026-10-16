import re
import glob
import os
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/b2_*.txt"
pat = re.compile(
    r"Results of task \[5\]: \[ class_inc_last_acc=([\d.]+)%.*?"
    r"class_inc_avg_acc=([\d.]+)%.*?forg=\s*([\d.]+)%",
    re.S,
)
size_re = re.compile(r"size=([\d.]+)MB")
bits_re = re.compile(r"weight_bits=(\[[^\]]*\])")

print(f"{'log':38s} {'last':>7s} {'avg':>7s} {'forg':>7s} {'size':>9s} w_bits")
for path in sorted(glob.glob(pattern)):
    text = open(path, encoding="utf-8", errors="replace").read()
    match = pat.search(text)
    sizes = size_re.findall(text)
    bits = bits_re.findall(text)
    if not match:
        print(f"{os.path.basename(path):38s} {'incomplete'}")
        continue
    print(
        f"{os.path.basename(path):38s} {match.group(1):>7s} {match.group(2):>7s} "
        f"{match.group(3):>7s} {(sizes[-1] if sizes else '-'):>9s} "
        f"{bits[-1] if bits else '-'}"
    )
