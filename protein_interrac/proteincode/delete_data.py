# encoding: utf-8
import os

f = open("A.txt")
lines = f.readlines()
count=0
for line in lines:
    if line[0].isupper():
        count+=1
        if line.__len__()<31:
            print count
        if "U" in line:
            print count/2+11
print count

#A
# delete        16763/16810/27235/27487/35251/36379/41572/48395/52717/56467/62015/63541/68979
# 对应的蛋白质序列 8382/8405 /13618/13744/17626/18190/20786/24198/26359/28234/31008/31771/34490

#B
# delete 276/327/379/1541/1906/2236/2372/2584/3238/3584/3626/3711/3862/4067/4234/4242/4737/5051/5458
#5482/5564/6601/6972/6974/7101/7152/7252/7316/7407/8074/8344/8463/8706/8832/9285/9401/9423/9869/9913/10026/10043/10089
#10487/10869/11290/11329/12663/12928/12950/13357/13458/13711/13827/13830/13877/13948/14211/14393/14724/14911/15089
#15244/15250/15329/15611/15641/15766/16253/16476/17107/17332/17379/17394/17491/17689/17707/17845/17913/18778/19178
#19854/20013/20189/20405/20622/20700/20864/20971/21164/21616/21874/22063/22424/22542/22561/23023/23346/23516/24122/24554
#24624/24707/25030/25044/25129/25639/25676/25829/26342/26404/26514/26725/27130/27250/27461/27586/27856/27975/28023/28024
#28340/28426/28619/28866/28931/29185/29261/29400/29799/29800/29942/30070/30380/31010/31017/31605/31744/32435/32622/33145
#33673/33771/34138/34165/34192/34294/34321/34351/34402/34441/34850/34889/35129/35289/35519/35727