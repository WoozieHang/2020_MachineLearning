import csv
import numpy as np
import math

def training():

    #读取数据
    with open('./train_set.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    #print(rows)  # 输出所有数据
    data_tmp = np.array(rows)  # rows是数据类型是‘list',转化为数组类型好处理
    data=np.ones((14000,17))
    for i in range(0,14000):
        for j in range(0,17):
            data[i][j]=float(data_tmp[i+1][j])

    #print("out0=", type(data), data.shape)
    #print("out1=", data)


    #OvR策略，26个循环，每个循环是一个二分类训练过程
    # w1,w2,...,w16,b共17个参数
    Beta = np.ones((26, 17)) * 0
    for i in range(0,26):
        #Beta^(t+1)=Beta^t+gamma*(sum{所有样本点}(x1;x2;...;x16;1)(y-e^(betaX)/（1+e^(betaX))))
        gama=0.000001
        # 反复迭代，牛顿梯度下降法求损失函数最小值
        for t in  range(0,500):
            grad = np.ones(17) * 0
            #对每个数据点，计算梯度的每一项，并求和
            for j in range(0,14000):
                grad_tmp=np.ones(17)*0
                for k in range(0,16):
                    grad_tmp[k]=data[j][k]
                grad_tmp[16]=1
                y=0
                #print(data[j][16],".",i+1)
                if data[j][16]==i+1:
                    y=1
                BetaX=0
                for k in range(0,16):
                    #print("i:",i,"j:",j,"k:",k,"t:",t)
                    #print("0:",Beta[i][k],",",data[j][k])
                    BetaX+=Beta[i][k]*data[j][k]
                BetaX+=Beta[i][16]
                #print(BetaX,";",j,";",t,";",y)
                #BetaX=round(BetaX,6)
                ExpBetaX=math.exp(BetaX)
                #ExpBetaX=1.0
                #print(ExpBetaX)
                grad_tmp=y-(ExpBetaX/(ExpBetaX+1))
                #print("grad_tmp:",grad_tmp)
                for k in range(0,16):
                    grad[k]=grad[k]+data[j][k]*grad_tmp
                grad[16]=grad[16]+grad_tmp
            Beta[i]=Beta[i]+np.dot(gama,grad)

            print("分类器号:",i+1,"/26; 迭代进度:",t+1,"/500")

    return Beta


def test(beta):
    rtn=np.ones((6000,26))
    # 读取数据
    with open('./test_set.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    # print(rows)  # 输出所有数据
    data_tmp = np.array(rows)  # rows是数据类型是‘list',转化为数组类型好处理
    data = np.ones((6000, 17))
    for i in range(0, 6000):
        for j in range(0, 17):
            data[i][j] = float(data_tmp[i + 1][j])

    #外层是26个分类测试循环，内层循环对每个测试样本，预测分类结果
    for i in range(0,26):
        for j in range(0,6000):
            #准确分类结果为
            #y=0
            #if data[j][16] == i + 1:
            #    y = 1
            #计算预测结果
            BetaX=0
            for k in range(0,16):
                BetaX+=beta[i][k]*data[j][k]
            BetaX+=beta[i][16]
            #否定概率
            #print(BetaX)
            guessYZero=1/(1+math.exp(BetaX))
            guessYTrue=1-guessYZero
            rtn[j][i]=guessYTrue
    #result保存着6000个测试点的26个二分类结果
    np.savetxt('./result.csv',rtn,fmt='%0.8f',delimiter=',')
    return

def analyse():
    #先处理得到6000个测试点的<预测值，真实值>pairs
    with open('./result.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    # print(rows)  # 输出所有数据
    data_tmp = np.array(rows)  # rows是数据类型是‘list',转化为数组类型好处理
    result = np.ones((6000, 26))
    for i in range(0, 6000):
        for j in range(0, 26):
            result[i][j] = float(data_tmp[i][j])
    #print(result)

    with open('./test_set.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    # print(rows)  # 输出所有数据
    data_tmp = np.array(rows)  # rows是数据类型是‘list',转化为数组类型好处理
    data = np.ones((6000, 17))
    for i in range(0, 6000):
        for j in range(0, 17):
            data[i][j] = float(data_tmp[i + 1][j])
    #print(data)

    out=np.zeros((6000,2))

    for i in range(0,6000):
        max=0
        max_value=result[i][0]
        for j in range(1,26):
            if result[i][j]>max_value:
                max=j
                max_value=result[i][j]

        out[i][0]=int(max+1)
        out[i][1]=int(data[i][16])

    #6000个测试点的 <预测值,真实值> pairs存到out.csv中
    np.savetxt('./out.csv', out,fmt='%0.1f', delimiter=',')

    #然后输出performance

    #accuracy
    right_num=0
    for i in range(0,6000):
        if out[i][0]==out[i][1]:
            right_num+=1

    accuracy=right_num/6000
    print("accuracy: ",accuracy)

    #26个分类器，每个分类器的TP,FP,FN,P,R,F1
    perf=np.zeros((26,6))*0
    for i in range(0,6000):
        if(out[i][0]==out[i][1]):
            #TP
            perf[int(out[i][0]-1)][0]+=1
        else:
            #FP
            perf[int(out[i][0]-1)][1]+=1
            #FN
            perf[int(out[i][1]-1)][2]+=1
    for i in range(0,26):
        #P
        if(not perf[i][0]+perf[i][1]==0):
            perf[i][3]=perf[i][0]/(perf[i][0]+perf[i][1])
        else:
            #print("1tag ",i)
            perf[i][3]=0
        #R
        if (not perf[i][0] + perf[i][2] == 0):
            perf[i][4]=perf[i][0]/(perf[i][0]+perf[i][2])
        else:
            #print("2tag ", i)
            perf[i][4]=0
        #f1
        if (not perf[i][3] + perf[i][4] == 0):
            perf[i][5]=2*perf[i][3]*perf[i][4]/(perf[i][3]+perf[i][4])
        else:
            #print("3tag ", i)
            perf[i][5]=0
    #micro precision;micro recall;micro F1
    tp_total=0
    fp_total=0
    fn_total=0
    for i in range(0,26):
        tp_total+=perf[i][0]
        fp_total+=perf[i][1]
        fn_total+=perf[i][2]

    microPrecision=tp_total/(tp_total+fp_total)
    microRecall=tp_total/(tp_total+fn_total)
    microF1=2*microPrecision*microRecall/(microPrecision+microRecall)
    print("microPrecision: ",microPrecision)
    print("microRecall: ", microRecall)
    print("microF1: ",microF1)

    #macro precision;macro recall;macro F1
    macroPrecision=0
    macroRecall=0
    macroF1=0
    for i in range(0,26):
        macroPrecision+=perf[i][3]
        macroRecall+=perf[i][4]
        macroF1+=perf[i][5]

    macroPrecision/=26
    macroRecall/=26
    macroF1/=26
    print("macroPrecision: ", macroPrecision)
    print("macroRecall: ", macroRecall)
    print("macroF1: ", macroF1)
    performance=np.zeros((7,1))*0
    performance[0][0]=accuracy
    performance[1][0]=microPrecision
    performance[2][0]=microRecall
    performance[3][0]=microF1
    performance[4][0] = macroPrecision
    performance[5][0] = macroRecall
    performance[6][0] = macroF1
    np.savetxt('./performance.csv', performance,fmt='%0.8f', delimiter=',')

if __name__ == '__main__':
    beta=training()
    test(beta)
    analyse()

