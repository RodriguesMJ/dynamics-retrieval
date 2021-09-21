# -*- coding: utf-8 -*-
import os

os.system('grep "Image filename" all_detwinned.stream > img_fn.txt')
os.system('grep "Event" all_detwinned.stream > event.txt')

img_fn = list(open('img_fn.txt', 'r'))
event_file = list(open('event.txt', 'r'))
n_events = len(img_fn)

out_fn = open('csplit.dat', 'w')
for i in range(n_events):
    print i
    if 'class1' in img_fn[i]:
        out_fn.write('%s %s class1\n'%(img_fn[i][16:].strip('\n'), event_file[i][9:].strip('\n')))
    elif 'class4' in img_fn[i]:
        out_fn.write('%s %s class4\n'%(img_fn[i][16:].strip('\n'), event_file[i][9:].strip('\n')))
    else:
        print 'issue'
out_fn.close()
