from shutil import move
import random
a= os.listdir('../data/')
for i in a:
    if i[0]=='.':
        continue
    direc = '../data/' + str(i)
    b=os.listdir(direc)
    for k in range(int(len(b)*0.2)):
        if not os.path.exists('../testing/'+str(i)):
            os.makedirs('../testing/'+str(i))
        val = int(random.random()*len(b));
            file = b[val]
            b.remove(file)
            move(direc + '/'+file, '../testing/'+str(i)+'/'+file)
