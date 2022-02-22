from idx2numpy import convert_from_file
import matplotlib.pyplot as plt
import os, sys
import json

os.chdir(os.path.dirname(sys.argv[0]))
imgs = convert_from_file("dataset/test-images.idx")
lbls = convert_from_file("dataset/test-labels.idx")

def _create_cls_map(textfile, out_file):
    map = {k: chr(int(v)) for k, v in [line.strip().split() for line in open(textfile).readlines()]}
    json.dump(map, open(out_file, "w"), indent=2)

def _load_cls_map():
    fp = "dataset/class-map.json"
    if not os.path.exists(fp):
        _create_cls_map("dataset/class-map.txt", fp)
    return {int(k): v for k, v in json.load(open(fp)).items()}

cls_map = _load_cls_map()

for i in range(1000, 1010):
    arr = imgs[i]
    print(cls_map[lbls[i]])
    plt.imshow(arr)
    plt.show()
