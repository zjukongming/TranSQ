import sys
import pickle as pkl
sys.path.append("..")
from transq.modules.gallery import Gallery


path = "/big-disk/kongming/TranSQ/iu_xray"
#save
sent_gallery = Gallery(path)
print(len(sent_gallery))
f = open("./data/iuxray_sentence_gallery.pkl", "wb")
pkl.dump(sent_gallery, f)