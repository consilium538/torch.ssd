import lxml.etree as et
from pathlib import Path
import pickle

cfg = {
    'year':'2007',
    'set':'trainval'
}

vocclasses = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
        )

basepath = Path(f'./VOCdevkit/VOC{cfg["year"]}')
setpath = basepath / 'ImageSets' / 'Main' / f'{cfg["set"]}.txt'
annopath = basepath / 'Annotations'
dumppath = Path(f'./anno_{cfg["year"]}_{cfg["set"]}.bin')

def classlabel(classname):
    return next(filter(lambda x: x[1] == classname, enumerate(vocclasses)))

def setname(setpath):
    with open(setfile, 'r') as fp:
        annolist = fp.readlines()
    return map(lambda x: x.strip(), annolist)

def annoparse(annopath):
    lxmltree = et.parse(annopath)
    filename = lxmltree.xpath('//filename/text()')[0]
    bboxlist = list()
    for i in lxmltree.xpath('//object'):
        bboxlist.append((
                    classlabel(i.xpath('name/text()')[0])
                    i.xpath('bndbox/xmin/text()')[0]
                    i.xpath('bndbox/ymin/text()')[0]
                    i.xpath('bndbox/xmax/text()')[0]
                    i.xpath('bndbox/ymax/text()')[0]
                ))
    return (fielname, bboxlist)

def anno_script():
    anno_list = list()
    for i in setname(setpath):
        anno_list.append(annoparse(annopath / f'{i}.xml'))
    with open(dumppath,'wb') as fp:
        pickle.dump(anno_list,fp)
    return

if __name__ == '__main__':
    anno_script()
