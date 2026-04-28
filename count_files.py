import os
folders = ['stained', 'unstained', 'tes_stain', 'tes_unstain']
total = 0
for f in folders:
    path = f"1000/{f}"
    if os.path.exists(path):
        count = len([x for x in os.listdir(path) if x.endswith('.tif')])
    else:
        count = 0
    print(f"{f}: {count}")
    total += count
print(f"Total: {total}")