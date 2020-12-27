import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
import math

# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

def inputData(filePath):
    # 根据文件路径，读入数据删除缺失行并对离散数据进行映射处理
    DataSet = pd.read_csv(filePath, header=None, skiprows=0,
                          names=["age", "workclass", "fnlwgt", "education",
                                 "education_num", "marital-status", "occupation", "relationship", "race", "sex",
                                 "capital-gain", "capital-loss",
                                 "hours-per-week", "native-country", "money"])

    DataSet.replace(' ?', np.nan, inplace=True)
    DataSet.dropna(axis=0, inplace=True)
    DataSet.reset_index(drop=True, inplace=True)
    # print(DataSet)

    for col in DataSet.columns[1::]:  # 循环修改数据类型的数组
        u = DataSet[col].unique()  # 将当前循环的数组加上索引

        def convert(x):
            return np.argwhere(u == x)[0, 0]  # 返回值是u中的数据等于x的索引数组，[0，0]截取索引数组的第一排第一个

        DataSet[col] = DataSet[col].map(convert)  # 索引替换映射数据

    return DataSet



class RandomForestModel:
    def __init__(self):
        # 初始化弱训练器列表
        self.h = []
        self.h_num = 0

        #初始化auc列表，用于5折交叉验证以保存输出综合结果
        self.aucList=[]

    def reStartMode(self):
        # 初始化弱训练器列表
        self.h = []
        self.h_num = 0

    def train(self,trainingDataSet,validDataSet,Epoch,discount_id,have_valid):
        self.reStartMode()

        # 初始化训练集
        D0 = trainingDataSet.copy()

        X = D0.iloc[:, 0:13]
        Y = D0.iloc[:, 14]

        # 训练轮次T
        T = Epoch
        sampleNum = len(D0)

        # T轮循环
        for t in range(0, T):
            # bootstrap
            D = D0.sample(n=sampleNum, replace=True)
            D.reset_index(drop=True, inplace=True)
            X = D.iloc[:, 0:13]
            Y = D.iloc[:, 14]

            # 训练弱训练器:随机决策树
            self.h.append(tree.DecisionTreeClassifier(criterion="entropy",max_features='log2'))
            self.h[t].fit(X, Y)
            # 训练损失
            e = 1 - self.h[t].score(X, Y)


            #使用验证集
            if have_valid==True:
                accur, auc = self.test(validDataSet)
                print("交叉验证进度:", str(discount_id) + "/5",
                      "   训练轮次:", str(1 + t) + "/" + str(T),
                      "   第" + str(1 + t) + "轮弱学习器训练集准确率：", 1 - e,
                      "   前" + str(1 + t) + "轮集成后验证集准确率：", accur,
                      "   前" + str(1 + t) + "轮集成后验证集auc-score:", auc)

                if t == len(self.aucList):
                    self.aucList.append(auc)
                else:
                    self.aucList[t] += auc
            else:
                accur, auc = self.test(validDataSet)
                print("用全部训练集进行正式训练",
                      "   训练轮次:", str(1 + t) + "/" + str(T),
                      "   第" + str(1 + t) + "轮弱学习器训练集准确率：", 1 - e,
                      "   前" + str(1 + t) + "轮集成后测试集准确率：", accur,
                      "   前" + str(1 + t) + "轮集成后测试集auc-score:", auc)


        #使用测试集
        if have_valid==False:
            accur, auc = self.test(validDataSet)
            print("测试集准确率:",accur,"测试集auc:",auc)

    def test(self,testDataSet):
        self.h_num = len(self.h)

        # 读入测试数据
        X_test = testDataSet.iloc[:, 0:13]
        Y_test = testDataSet.iloc[:, 14]

        # 预测
        Y_pred = self.h[0].predict(X_test)
        for i in range(1, self.h_num):
            Y_pred = Y_pred + self.h[i].predict(X_test)

        for i in range(0, len(Y_pred)):
             Y_pred[i]=round(Y_pred[i]/self.h_num)

        # print(Y_pred)
        # 测试报告
        #print(classification_report(Y_test, Y_pred))

        # 测试auc
        accur=accuracy_score(Y_test,Y_pred)
        auc=roc_auc_score(Y_test, Y_pred)

        return accur, auc

    def ouputBestEpochByAuc(self,discount):
        for i in range(0,len(self.aucList)):
            #计算平均
            self.aucList[i]/=discount
        # print(self.aucList)

        max_id=0
        max=self.aucList[0]

        for i in range(1,len(self.aucList)):
            if self.aucList[i]>max:
                max_id=i
                max=self.aucList[i]

        print("经过五折交叉验证, auc最佳的训练轮次设置为",max_id+1,"  auc大小为",max)

        # #epoch-auc 图像显示
        # xValue = [i for i in range(1,len(self.aucList)+1)]
        # yValue = self.aucList.copy()
        #
        # z1 = np.polyfit(xValue, yValue, 11)  # 用4次多项式拟合
        # p1 = np.poly1d(z1)
        #
        # y_vals = p1(xValue)  # 也可以使用yvals=np.polyval(z1,x)
        #
        # plt.title(u'Validation AUC-SCORES vs Base Learner Number for RandomForest', FontProperties=font)
        #
        # plt.xlabel('epochs')
        # plt.ylabel('AUC-SCORES')
        # # plt.scatter(x, y, s, c, marker)
        # # x: x轴坐标
        # # y：y轴坐标
        # # s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
        # # c: 点的颜色
        # # marker: 标记的样式 默认是 'o'
        # plt.legend()
        # # plt.plot(xValue, yValue, s=20, c="#ff1212", marker='o')
        # plot1 = plt.plot(xValue, yValue, 'o', label='original values')
        # plot2 = plt.plot(xValue, y_vals, 'r', label='polyfit values')
        # # plt.scatter(xValue, yValue, s=20, c="#ff1212", marker='o')
        # plt.show()

        return max_id+1


#主程序开始位置


# 读入训练数据
trainingDataSet=inputData('./adult.data')

# 读入测试数据
testDataSet=inputData('./adult.test')

#   方案1：使用自带的训练集和测试集
# pass

#   方案2：合并训练和测试数据，并重划分四分之一出作为测试集，因为感觉其自带的训练集和测试集不是同分布，故采取这样的策略保证同分布
# totalDataSet=pd.concat([trainingDataSet,testDataSet]).copy()
# trainingDataSet, testDataSet = train_test_split(totalDataSet,test_size=0.25,random_state=42)

#   方案3：丢弃原先测试集，在自带的训练集中划分四分之一作为测试集，剩余的四分之三作为新的训练集。此乃最终方案
trainingDataSet, testDataSet = train_test_split(trainingDataSet,test_size=0.25,random_state=42)

#设置超参数轮次
epoch=100

#划分验证集5个
validDataSet1,validDataSet2=train_test_split(trainingDataSet,test_size=0.2,random_state=42)
validDataSet1,validDataSet3=train_test_split(validDataSet1,test_size=0.25,random_state=42)
validDataSet1,validDataSet4=train_test_split(validDataSet1,test_size=1/3,random_state=42)
validDataSet1,validDataSet5=train_test_split(validDataSet1,test_size=0.5,random_state=42)

#组合验证集得到训练集对应5个
trainingDataSet1=pd.concat([validDataSet2,validDataSet3,validDataSet4,validDataSet5]).copy()
trainingDataSet2=pd.concat([validDataSet1,validDataSet3,validDataSet4,validDataSet5]).copy()
trainingDataSet3=pd.concat([validDataSet1,validDataSet2,validDataSet4,validDataSet5]).copy()
trainingDataSet4=pd.concat([validDataSet1,validDataSet2,validDataSet3,validDataSet5]).copy()
trainingDataSet5=pd.concat([validDataSet1,validDataSet2,validDataSet3,validDataSet4]).copy()

#重置索引
trainingDataSet.reset_index(drop=True, inplace=True)
trainingDataSet1.reset_index(drop=True, inplace=True)
trainingDataSet2.reset_index(drop=True, inplace=True)
trainingDataSet3.reset_index(drop=True, inplace=True)
trainingDataSet4.reset_index(drop=True, inplace=True)
trainingDataSet5.reset_index(drop=True, inplace=True)

validDataSet1.reset_index(drop=True, inplace=True)
validDataSet2.reset_index(drop=True, inplace=True)
validDataSet3.reset_index(drop=True, inplace=True)
validDataSet4.reset_index(drop=True, inplace=True)
validDataSet5.reset_index(drop=True, inplace=True)

testDataSet.reset_index(drop=True,inplace=True)

abm=RandomForestModel()

#五折交叉验证
abm.train(trainingDataSet1,validDataSet1,epoch,1,True)
abm.train(trainingDataSet2, validDataSet2,epoch,2,True)
abm.train(trainingDataSet3, validDataSet3,epoch,3,True)
abm.train(trainingDataSet4, validDataSet4,epoch,4,True)
abm.train(trainingDataSet5, validDataSet5,epoch,5,True)
BestEpoch=abm.ouputBestEpochByAuc(5)

#训练并输出测试结果
abm.train(trainingDataSet,testDataSet,BestEpoch,0,False)
