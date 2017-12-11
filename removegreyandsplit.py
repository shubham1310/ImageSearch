import os
from shutil import move
import random
from scipy.misc import imread

d='../256_ObjectCategories/'
a=os.listdir(d)
for i in a:    
    b=os.path.join(d,i)
    c=os.listdir(b)
    for j in c:
        if(len(imread(os.path.join(b,j)).shape)<3):
            os.remove(os.path.join(b,j))
            print(b,j)

dire= '256_ObjectCategories'
a= os.listdir('../'+ dire+'/')
a.sort()
for i in a:
    if i[0]=='.':
        continue
    # print(i)
    direc = '../'+ dire+'/' + str(i)
    b=os.listdir(direc)
    for k in range(int(len(b)*0.2)):
        # print(i)
        if not os.path.exists('./testing/'+str(i)):
            os.makedirs('./testing/'+str(i))
        val = int(random.random()*len(b));
        file = b[val]
        b.remove(file)
        print('./testing/'+str(i)+'/'+file)
        move(direc + '/'+file, './testing/'+str(i)+'/'+file)
