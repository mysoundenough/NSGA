# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:00:52 2022

@author: Mayuan

简介：采用经营选择遗传算法进行JOBSHOP作业顺序优化
"""


import numpy as np
import pandas as pd
import time


'''更换问题时更改''' 
JOB = 10
MACHINE = 4
JIYIN = 40
MUBIAO = 746.83 # 目标时间（最优>716.47）
m = 1 # 染色体条数(更改后代码改动较大，尽量不要更改)



infor = np.zeros((JOB, 2*MACHINE), dtype=float) # 加工时间与工序顺序数据
workindex = np.zeros((JOB), dtype=int)          # 存储基因范围所代表的加工零件

# 读取数据并初始化workindex
def inputInformation():
    data = pd.read_csv(r'./data/data.csv', encoding='gbk')
    data = np.array(data)
    for i in range(JOB):
        for j in range(2*MACHINE):
            infor[i][j] = data[i][j]
    # 初始化workindex
    for i in range(JOB):
        for j in range(MACHINE):
            if (infor[i][2*j] >= 0):
                workindex[i] += 1
        if (i < JOB-1):
            workindex[i + 1] = workindex[i]

# inputInformation()


# 交换两个数值
def swap(a, b):
    temp = a
    a = b
    b = temp
    return a, b

# 获取随机序列(交换顺序随机)
def randArray(a):
    for i in range(JIYIN):
        a[i] = i
    for i in range(JIYIN):
        rd = np.random.randint(0,JIYIN-i) + i
        temp = swap(a[i], a[rd])
        a[i] = temp[0]
        a[rd] = temp[1]
    return a




'''GA方法初始化变量'''
GROUPSCALE = 6         # 初始种群规模(偶数)
SONNUM = 6             # 儿子种群规模(偶数)
inheritance = 40000       # 迭代次数

# NSGA将种群分为X个等级，GROUPSCALE/X必须为整数，且大于1，才能发挥NSGA的作用
LEVEL_NUM = 3

# 变异概率（可以动态，先大再小）
init_Mutation = 0.98   # 初始变异率
delta_Mutation = 0.18  # 变异率最大减少量

# 交叉概率（可以动态，先小再大）（大的容易跳出局部最优，但是模型不易收敛）
init_Crossover = 0.02   # 初始交叉率
delta_Crossover = 0.04  # 交叉率最大增量


'''运行记录---------------------------------------------------------------(18最优 = 746.83)----------
1.（20 20 15000 0.98 0.02 6000代稳定，出现了784.07，已保存在1（2）csv中）
2.(2 2 12000 1 0.25 最小时间：784.07，平均时间：810.172，最大代数：4709)
3.(2 2 6000  1 0.25 最小时间：827.81，平均时间：847.976，最大代数：4581)
4.(2 2 12000 1 0.35 最小时间：806.06，平均时间：831.612，最大代数：5449)
5.(2 2 12000 1 0.20 最小时间：784.57，平均时间：835.29， 最大代数：11992)
6.(2 2 20000 1 0.20 最小时间：781.98，平均时间：808.072，最大代数：6675)
7.(2 2 20000 1 0.15 最小时间：805.83，平均时间：829.518，最大代数：17082)
8.(2 2 20000 1 0.15 最小时间：762.7， 平均时间：830.4，  最大代数：10960，平均代数：7165.2)
9.(2 2 20000 1 0.15 最小时间：802.89，平均时间：823.502，最大代数：17622，平均代数：6016.4)
1.(2 2 20000 1 0.18 最小时间：771.87，平均时间：795.822，最大代数：8421，平均代数：5311.4)
2.(2 2 20000 1 0.18 最小时间：794.41，平均时间：831.639，最大代数：12612，平均代数：4122.5)
2.(2 2 20000 1 0.17 最小时间：778.29，平均时间：810.308，最大代数：10020，平均代数：6017.2
2.(2 2 20000 1 0.16 最小时间：789.14，平均时间：830.412，最大代数：8402，平均代数：5493.6
2.(2 2 20000 1 0.15 最小时间：773.77，平均时间：827.986，最大代数：9374，平均代数：6636.6
2.(2 2 20000 1 0.14 最小时间：794.31，平均时间：807.092，最大代数：11139，平均代数：6679.4
2.(2 2 20000 1 0.13 最小时间：785.74，平均时间：832.738，最大代数：19174，平均代数：11485.6
2.(2 2 20000 1 0.12 最小时间：797.07，平均时间：835.21， 最大代数：10714，平均代数：7574.0
2.(2 2 20000 1 0.11 最小时间：773.77，平均时间：806.936，最大代数：19857，平均代数：10458.0

   
最优参数：
(2 2 20000 0.94 0.11 最小时间：779.16，平均时间：797.196，最大代数：19802，平均代数：10968.6
(2 2 20000 0.91 0.13 最小时间：773.77，平均时间：801.168，最大代数：13637，平均代数：10077.4
(2 2 20000 0.92 0.10 最小时间：787.42，平均时间：808.332，最大代数：5172，平均代数：4120.6

------------------------------------------------------------------------------  引入动态率---------
2.（20 20 5000 0.98-0.8 0.02-0.06 2000代稳定，多次出现850左右，未保存）
3.（50 50 5000 0.98-0.8 0.02-0.06 1000代稳定，多次出现900左右，未保存）
4.（5000 5000 5000 0.98-0.8 0.02-0.06 时间太长）
5.（100 100 5000 0.98-0.8 0.02-0.06 1000代稳定，多次出现900左右，未保存）
6.（10 10 5000 0.98-0.8 0.02-0.06 3000代稳定，多次出现820左右，未保存）
7.（2 2 20000 0.98-0.8 0.02-0.06 17000代稳定，多次出现800左右，出现778.47，已保存在1（2）csv中）

目前最优：
8.（2 2 40000 0.98-0.8 0.02-0.06 30000代稳定，多次出现780以下，出现773.77，已保存在1（2）csv中）

9.（2 2 40000 0.98-0.90 0.02-0.04 40000代稳定，多次出现800左右，出现779，未保存）
10.（2 2 60000 0.98-0.90 0.02-0.04,50000代稳定，多次出现830左右，出现785，未保存）
11.（2 2 80000 0.98-0.70 0.02-0.10,30000代稳定，多次出现820左右，出现798，未保存）
12.（2 2 20000 0.98-0.70 0.02-0.10,20000代稳定，多次出现800以下，出现788，未保存）
13.（2 2 40000 0.98-0.70 0.02-0.10,30000代稳定，多次出现790以下，未保存）
14.采用最优优化一晚:
（2 2 40000 0.98-0.8 0.02-0.06 33000代稳定，多次出现780以下，出现766.80，已保存在1（2）csv中）
15.采用最优优化10分钟不到就出现了766.32，已保存
16.采用最优优化10分钟不到就出现了762.70，已保存
17.采用最优优化30分钟不到就出现了760.39，已保存
18.采用最优优化30分钟不到就出现了746.83，已保存
19.（2 2 40000 0.98-0.96 0.02-0.04 30000代稳定，多次出现780以下，出现773.77，已保存在1（2）csv中）
20.（2 2 2000 0.98-0.96 0.5-0.8 30000代稳定，多次出现780以下，出现773.77，已保存在1（2）csv中）
21.（2 2 3000 0.05-0.03 0.5-0.6 效果不好）
22.（2 2 3000 0.5-0.3 0.5-0.6 出现820左右）
23.（2 2 3000 0.5-0.3 0.5-0.6 出现820左右）
24.（4 4 3000 0.5-0.3 0.5-0.6 出现870左右）
25.（4 4 3000 0.5-0.3 0.5-0.6 出现850左右）
26.（8 8 3000 0.5-0.3 0.5-0.6 出现820左右）
27.（16 16 3000 0.5-0.3 0.5-0.6 出现800左右）
28.（16 16 1000 0.5-0.5 0.5-0.5 出现800多）
29.（16 16 1000 0.8-0.8 0.5-0.5 出现800多）


-----------------------------------------------------------------------初始种群引入历代个体------
1.引入最优两个（8 8 40000 0.98-0.8 0.5-0.8 均未出现746.83以下）
2.引入较优两个（8 8 40000 0.98-0.8 0.5-0.8 均未出现最小以下）
3.引入较优两个（4 4 40000 0.98-0.8 0.5-0.8 均未出现最小以下）
4.引入较优两个（4 4 40000 0.98-0.8 0.9-0.99 均未出现最小以下）

'''


# 个体类定义
class Unit:
    def __init__(self):
        self.chromosome = np.zeros((1,JIYIN), dtype=int)
        self.fitness = 0
        self.proba = 0

# 种群字典定义
group = {}      # 初始种群
sonGroup = {}   # 儿子种群
for i in range(GROUPSCALE):
    group['obj'+str(i)] = Unit()
    sonGroup['obj'+str(i)] = Unit()
newbornNum = 0      # 新生儿子个数
indexcross_i = 0    # 即将交叉染色体片段起始位置
indexcross_j = 0    # 即将交叉染色体片段终止位置


# 输出个体
def printUnit(unit):
    print('\n基因型：', unit.chromosome)
    print('\n适应度：', unit.fitness)
    print('\n选为父本率：', unit.proba)

# printUnit(group['obj'+str(1)])

# 输出个体代表的加工过程
def printProcess(unit):
    numjob = 0          # 加工到第几个工件
    orderprocess = np.zeros((JIYIN), dtype=int)   # 加工顺序
    for i in range(JIYIN):
        for j in range(JOB):
            if (j == 0):
                if (0 <= unit.chromosome[0][i] and unit.chromosome[0][i] < workindex[j]):
                    numjob = j
            else :
                if (workindex[j - 1] <= unit.chromosome[0][i] and unit.chromosome[0][i] < workindex[j]):
                    numjob = j
        orderprocess[i] = numjob
    print('\n加工顺序：{}\n\n'.format(orderprocess))

# printProcess(group['obj'+str(1)])


# 时间计算(1：1-n)，该问题中，即为加工这些工件的时间花费
def calculateTime_GA(unit):
    maxtime = 0.0       # 求出加工时间
    starttime = 0.0
    endtime = 0.0
    worktime = 0.0
    numjob = 0         # 加工到第几个工件 0-8
    nummachine = 0     # 加工到第几个机器 0-3
    numwork = np.zeros((JOB), dtype=int)           # 第n个工件加工到第几个工序
    machinetime= np.zeros((MACHINE), dtype=float)    # 存储机器m（0-3）的空闲时间点
    jobtime = np.zeros((JOB), dtype=float)           # 存储工件上一道工序结束时间
    # 对序列进行遍历
    for i in range(JIYIN):
        # 确定序号属于哪个作业的，并确定工序号
        for j in range(JOB):
            if (j == 0):
                if (0 <= unit.chromosome[0][i] and unit.chromosome[0][i] < workindex[j]):
                    numjob = j
                    numwork[numjob] = numwork[numjob] + 1 #1-4
            else :
                if (workindex[j - 1] <= unit.chromosome[0][i] and unit.chromosome[0][i] < workindex[j]):
                    numjob = j
                    numwork[numjob] = numwork[numjob] + 1
        # 查找infor表，确定机器编号和加工时间
        for j in range(MACHINE):
            if ((numwork[numjob] - 1) == infor[numjob][2*j]):
                nummachine = j - 1 # 0-3 (j-1)
                worktime = infor[numjob][2*j + 1]
        # (动态规划)比较机器空闲时间点和上一道工序完成时间对该工序完成时间进行计算并存储
        starttime = max(machinetime[nummachine], jobtime[numjob])
        endtime = starttime + worktime
        jobtime[numjob] = endtime
        machinetime[nummachine] = endtime
    # 求出JOB个作业的完成时间的最大值，返回完成时间
    for i in range(MACHINE):
        if (machinetime[i] > maxtime):
            maxtime = machinetime[i]
            
    return round(maxtime, 2)

                    
# 计算种群各个个体被选择作为父本概率，传入的数为种群fitness之和
def calculateProbality(total):
    tempTotalP = 0.0
    for i in range(GROUPSCALE):
        # 个体概率=种群总时间/个体时间，（某个体加工时间越短，做父本概率越大）
        group['obj'+str(i)].proba = (1.0 / float(group['obj'+str(i)].fitness)) * float(total)
        tempTotalP += group['obj'+str(i)].proba
    for i in range(GROUPSCALE):
        group['obj'+str(i)].proba = group['obj'+str(i)].proba / tempTotalP


# 初始化种群方式1.随机初始化
def init_GA():
    total = 0
    for i in range(GROUPSCALE):
        # 基因型初始化
        group['obj'+str(i)].chromosome[0] = randArray(group['obj'+str(i)].chromosome[0])
        # 适应度初始化
        group['obj'+str(i)].fitness = calculateTime_GA(group['obj'+str(i)])
        # 种群适应度之和
        total += group['obj'+str(i)].fitness   
    # 选为父本概率初始化
    calculateProbality(total)
    # 种群排序
    SortGroup()

# 初始化种群方式2.选取最优初始化
def init_GA2():
    total = 0
    for i in range(GROUPSCALE):
        # 基因型初始化
        if (i == 0):
            group['obj'+str(i)].chromosome[0] = [14,31,12,3,11,15,27,13,1,18,0,30,38,34,8,29,19,2,9,39,10,4,17,25,36,33,24,35,28,16,22,37,7,6,32,5,26,21,20,23]
        elif (i == 1):
            group['obj'+str(i)].chromosome[0] = [14,37,12,30,36,29,4,1,9,20,13,7,38,3,8,34,6,16,15,21,39,31,11,35,24,2,19,5,23,33,28,10,0,18,22,26,32,17,27,25]
        else :
            group['obj'+str(i)].chromosome[0] = randArray(group['obj'+str(i)].chromosome[0])
        # 适应度初始化
        group['obj'+str(i)].fitness = calculateTime_GA(group['obj'+str(i)])
        # 种群适应度之和
        total += group['obj'+str(i)].fitness   
    # 选为父本概率初始化
    calculateProbality(total)


# init_GA()
# printUnit(group['obj'+str(0)])
# printProcess(group['obj'+str(0)])



# 挑选父本方式1：精英选择，直接选择种群中适应度最小的两个个体
def selectTwo(father_pos, mother_pos):
    one = 0
    minCost = float('inf')
    for i in range(GROUPSCALE):
        if (minCost > group['obj'+str(i)].fitness):
            minCost = group['obj'+str(i)].fitness
            one = i
    father_pos = one
    two = 0
    minCost = float('inf')
    for i in range(GROUPSCALE):
        if (i == father_pos):
            continue
        if (minCost > group['obj'+str(i)].fitness):
            minCost = group['obj'+str(i)].fitness
            two = i
    mother_pos = two
    return one, two

# 挑选父本方式2：轮盘赌选择，模拟自然选择，依据被选择概率随机选择两个个体做父本
def selectOne(conf):
    selectP = float(np.random.randint(0,999)/998.0)
    sumP = 0.0
    for i in range(GROUPSCALE):
        sumP += group['obj'+str(i)].proba
        if (selectP < sumP):
            if (i == conf and i != GROUPSCALE - 1): 
                return i + 1 # 解决选取的两个父本是同一个的问题
            else :
                return i
    return 0


# 挑选父本方式3：轮盘赌+精英选择（NSGA），将种群个体按照适应度分级，适应度越高级别越高，级别越高的群体被选择概率越大
def selectThree(conf):
    
    # 1、现对种群的是适应度排序，适应度由高到低排序到变量中
    # 11、创建一个排序矩阵，第一行存储适应度（时间），第二行存储0至种群规模-1
    fitness_sort = np.zeros((3,GROUPSCALE), dtype=float)
    for i in range(GROUPSCALE):
        fitness_sort[0,i] = group['obj'+str(i)].fitness
        fitness_sort[1,i] = i
    f1 = np.lexsort(fitness_sort[::-1,:])
    Fitness_sort = fitness_sort.T[f1].T
    
    # 2、跟据种群大小，将种群分为X个等级，每个等级内赋予一个相同的虚拟适应度，等级越高适应度越高
    # 计算每个等级有几个个体，LEVEL_NUM 为超参数
    levelscale = int(GROUPSCALE/LEVEL_NUM)
    sum_Fitness_sort = 0.0
    for i in range(GROUPSCALE):
        Fitness_sort[2,i] = group['obj'+str(levelscale*int(np.floor(i/levelscale)))].proba
        sum_Fitness_sort += Fitness_sort[2,i]
    for i in range(GROUPSCALE):
        Fitness_sort[2,i] = Fitness_sort[2,i] / sum_Fitness_sort
        
    # 3、根据虚拟适应度开始轮盘赌挑选个体
    selectP = float(np.random.randint(0,999)/998.0)
    sumP = 0.0
    for i in range(GROUPSCALE):
        sumP += Fitness_sort[2,i]
        if (selectP < sumP):
            if (i == conf and i != GROUPSCALE - 1): 
                return i + 1 # 解决选取的两个父本是同一个的问题
            else :
                return i
    return 0



# 找到基因冲突
#（交叉后与原来重复的染色体就是冲突，染色体编码规则不允许该重复，会破坏计算）
def getConflict(Detection, Model, len_cross, len_conflict, conflict):
    len_conflict = 0
    for i in range(len_cross):
        flag = 1
        for j in range(len_cross):
            if (Detection[i] == Model[j]):
                j = len_cross
                flag = 0 # 如果交叉片段存在重复，交叉后不冲突，因此这个位置不是冲突
        if (flag):
            conflict[len_conflict] = Detection[i]
            len_conflict += 1
    return len_conflict, conflict


# 解决冲突，使得一个排列中不出现重复的编号
# （解决方法：将父（母）交叉片段冲突部分，替换到与母（父）未交叉片段中）
def handleConflict(conflictUnit, Detection_Conflict, Model_Conflict, len_conflict, k, p):
    for i in range(len_conflict):
        flag = 0
        index = 0
        # 重复的编号在Model_Conflict中
        # 以下对非交换区域找出编号重复的位置
        for index in range(indexcross_i):
            if (Model_Conflict[i] == conflictUnit.chromosome[k][index]):
                flag = 1 # 如果基因重复了，就不需要交换
                break
        if (flag == 0):
            for index in range(indexcross_j + 1, JIYIN):
                if (Model_Conflict[i] == conflictUnit.chromosome[k][index]):
                    break
        # 将重复的编号替换，替换编号在Detection_Conflict中
        conflictUnit.chromosome[k][index] = Detection_Conflict[i]
    
    # 终止将这条新染色体赋值，传给孩子
    for i in range(JIYIN):
        p[i] = conflictUnit.chromosome[k][i]
    return Detection_Conflict, Model_Conflict, p
    

# 采用PMX方式进行交叉变异
def Crossover_PMX(fa, mo):
    
    son_one = Unit()
    son_two = Unit()
    son_one.fitness = 0
    son_two.fitness = 0
    
    k = 0
    
    for k in range(m):
        # 交叉变异初始与结束位置
        indexcross_i = np.random.randint(0,JIYIN)
        indexcross_j = np.random.randint(0,JIYIN)
        if (indexcross_i > indexcross_j):
            temp = indexcross_i
            indexcross_i = indexcross_j
            indexcross_j = temp
        
        father_cross = np.zeros((JIYIN), dtype=int) # 记录父亲交叉片段
        mother_cross = np.zeros((JIYIN), dtype=int) # 记录母亲交叉片段
        len_cross = 0
        for i in range(indexcross_i, indexcross_j):
            father_cross[len_cross] = fa.chromosome[k][i]
            mother_cross[len_cross] = mo.chromosome[k][i]
            len_cross += 1;
            
        conflict_fa = np.zeros((JIYIN), dtype=int) # 记录父亲产生的冲突片段
        conflict_ma = np.zeros((JIYIN), dtype=int) # 记录母亲产生的冲突片段
        len_conflict = 0 # 冲突基因个数
        len_conflict, conflict_fa = getConflict(father_cross, mother_cross, len_cross, len_conflict, conflict_fa)
        len_conflict, conflict_ma = getConflict(mother_cross, father_cross, len_cross, len_conflict, conflict_ma)
        
        for i in range(indexcross_i, indexcross_j):
            temp_node = fa.chromosome[k][i]
            fa.chromosome[k][i] = mo.chromosome[k][i]
            mo.chromosome[k][i] = temp_node
            
        conflict_fa, conflict_ma, son_one.chromosome[k] = handleConflict(fa, conflict_fa, conflict_ma, len_conflict, k, son_one.chromosome[k])
        conflict_ma, conflict_fa, son_two.chromosome[k] = handleConflict(mo, conflict_ma, conflict_fa, len_conflict, k, son_two.chromosome[k])
    
    # 将两个孩子加入子代种群
    global newbornNum
    sonGroup['obj'+str(newbornNum)] = son_one
    newbornNum = newbornNum + 1
    sonGroup['obj'+str(newbornNum)] = son_two
    newbornNum = newbornNum + 1
    return newbornNum


# 对第index个个体的第k条染色体进行变异    
def Mutation(index, k):
    gen_i = np.random.randint(0,JIYIN)
    gen_j = np.random.randint(0,JIYIN)
    # 变异采取的方式是随机交换两个基因
    temp = sonGroup['obj'+str(index)].chromosome[k][gen_i]
    sonGroup['obj'+str(index)].chromosome[k][gen_i] = sonGroup['obj'+str(index)].chromosome[k][gen_j]
    sonGroup['obj'+str(index)].chromosome[k][gen_j] = temp


# 种群更新
def UpdateGroup():
    
    tempP = Unit()
    
    # 对新生种群按适应度从小到大排序
    for i in range(newbornNum):
        for j in range(newbornNum - i - 1):
            j = newbornNum - j - 1
            if (sonGroup['obj'+str(i)].fitness > sonGroup['obj'+str(j)].fitness):
                tempP = sonGroup['obj'+str(i)]
                sonGroup['obj'+str(i)] = sonGroup['obj'+str(j)]
                sonGroup['obj'+str(j)] = tempP
                
    # 对原始种群按适应度从小到大排序
    for i in range(GROUPSCALE):
        for j in range(GROUPSCALE - i - 1):
            j = GROUPSCALE - j - 1
            if (group['obj'+str(i)].fitness > group['obj'+str(j)].fitness):
                tempP = group['obj'+str(i)]
                group['obj'+str(i)] = group['obj'+str(j)]
                group['obj'+str(j)] = tempP
                
    # 将新生种群中fitness较小的个体替换至原始种群，从而得到新种群
    j = GROUPSCALE - 1
    for i in range(newbornNum):
        if (sonGroup['obj'+str(i)].fitness < group['obj'+str(j)].fitness):
            group['obj'+str(j)] = sonGroup['obj'+str(i)];
            j -= 1
        else :
            break
    
    #计算新种群所以个体适应度之和
    total = 0
    for i in range(GROUPSCALE):
        total += group['obj'+str(i)].fitness
    return total


# 按照适应度大小对种群进行排序
def SortGroup():
    
    for i in range(GROUPSCALE):
        for j in range(GROUPSCALE - i - 1):
            j = GROUPSCALE - j - 1
            if (group['obj'+str(i)].fitness > group['obj'+str(j)].fitness):
                tempP = group['obj'+str(i)]
                group['obj'+str(i)] = group['obj'+str(j)]
                group['obj'+str(j)] = tempP



# 算法主要步骤
def GA():
    
    print("\n----------------------------------GA----------------------------------\n")
    
    init_GA()
    
    iter = 0
    minFitness = float('inf')
    minGeneration = float('inf')
    minUnit = Unit()
    while (iter < inheritance):
        
        #每次交叉产生两个儿子，故只需M次交叉
        M = GROUPSCALE - GROUPSCALE / 2
        global newbornNum
        newbornNum = 0 # 记录新生儿个数
        while (M) :
            # 1.选择，两种选择方式
            # (1)轮盘赌选择
            #pos1 = selectOne(-1)
            #pos2 = selectOne(pos1)
            # (2)精英选择
            # pos1 = 0
            # pos2 = 0
            # pos1, pos2 = selectTwo(pos1,pos2)
            # (3)精英+轮盘选择
            pos1 = selectThree(-1)
            pos2 = selectThree(pos1)
            
            # 确定两个父本个体
            father = group['obj'+str(pos1)]
            mother = group['obj'+str(pos2)]
            
            # 2.交叉
            Is_crossover = float(np.random.randint(0, 999) / 998.0)
            # 动态交叉率
            crossover = init_Crossover + delta_Crossover * (float(iter) / float(inheritance))
            if (Is_crossover <= crossover):
                newbornNum = Crossover_PMX(father, mother)
            
            M -= 1
            
        # 3.变异
        for i in range(newbornNum):
            for k in range(m):
                rateVaration = float(np.random.randint(0, 999) / 998.0)
                # 动态变异率
                mutation = init_Mutation - delta_Mutation * (float(iter) / float(inheritance))
                if (rateVaration < mutation):
                    Mutation(i, k)
            # 计算新生儿适应度
            sonGroup['obj'+str(i)].fitness = calculateTime_GA(sonGroup['obj'+str(i)])
        
        # 4.更新
        totaltime = UpdateGroup()       # 更新种群并得到新种群个体适应度之和
        SortGroup()                     # 更新时子种群较优个体插入了原种群，需要重新排序
        calculateProbality(totaltime)   # 计算新种群个体被选择概率
        
        iter += 1
        
        minCost = float('inf')
        pos  = float('inf')
        for i in range(GROUPSCALE):
            if (minCost > group['obj'+str(i)].fitness):
                minCost = group['obj'+str(i)].fitness
                pos = i
        
        # 记录历代最优
        if (minFitness > minCost):
            minFitness = minCost
            minUnit = group['obj'+str(pos)]
            minGeneration = iter
        # 每1000代打印当前迭代信息
        if (iter % 1000 == 0):
            print('当前{}/({})代 最优：{}代 时间：{} s'.format(iter, inheritance, minGeneration, minUnit.fitness))
    
    #5.输出最优    
    print("\n\nBest, No.{} generation:\n".format(minGeneration))
    printUnit(minUnit)
    printProcess(minUnit)
    return minGeneration, minUnit.fitness





# 多次运行遗传算法，统计运行数据
def main():
    
    start_time = time.time()
    
    global workindex
    workindex = np.zeros((JOB), dtype=int)
    inputInformation()
    
    # 统计多次跑的最优代与最优值以及平均值
    minf = 0
    gener = 0
    allMinf = []
    allGeneration = []
    
    '''优化方向
    1.保存历代最优个体，初始化种群时加入历代最优几个个体
    2.动态交叉率，先较小，找到局部最优，再调大，跳出局部最优
    '''
    
    # 总计最大运行次数（想得到最优，就用optimizer_rate找到最优参数，然后设置值足够跑一晚上）
    for i in range(2):
        
        print('\n第{}次运行遗传算法'.format(i+1))
        
        gener, minf = GA()
        allMinf.append(minf)
        allGeneration.append(gener)
        
        # 找到理想的值，停止计算
        if (minf <= MUBIAO):
            break
    
    print('共运行GA算法{}次：最小时间：{}，平均时间：{}，最大代数：{}，平均代数：{}\n'.format(i+1, min(allMinf), sum(allMinf)/len(allMinf), max(allGeneration), sum(allGeneration)/len(allGeneration)))
    
    end_time = time.time()
    
    print('算法用时：{}s'.format(end_time - start_time))



'''率参数优化
def optimize_rate():
    
    global init_Mutation
    global delta_Mutation
    global init_Crossover
    global delta_Crossover
    
    for p in range(10):
        for q in range(10):
            
            init_Mutation = 1 - 0.01 * p
            init_Crossover = 0.1 + 0.01 * q
            
            print('\n\n变异率：{}，交叉率：{}'.format(init_Mutation, init_Crossover))
            
            main()
'''  


    
main()