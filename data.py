from shutil import copyfile
i=0
src = 'oxbuild_images/'
for j in files:
    if 'good' in j:
        directory = 'newdata/'+ str(i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        a= open('gt_files_170407/'+j)
        k=0
        for b in a.read().split('\n'):
            if b=='':
                 continue
            else:
                 copyfile(src + b + '.jpg', directory +'/' + str(k) + '.jpg')
                 k+=1
                 print(i)
            i+=1