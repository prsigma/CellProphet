# -*- coding: utf-8 -*-

import numpy as np
import copy
import scipy.linalg
from pandas import DataFrame
import sys
import argparse
import warnings
import progressbar
import matplotlib.pyplot as plt

# hide warnings
warnings.filterwarnings('ignore')


class PriorBoost:
    def __init__(self, ExpFile, Net1File, Net2File):
        self.xi = 1.
        self.mu = 1.
        # read in Exp
        self.ReadinExp(ExpFile)
        # read in Net1
        self.ReadinNet1(Net1File)
        # read in Net2
        self.ReadinNet2(Net2File)
        self.A = np.matrix(np.zeros((self.NumTF, self.NumExp)))
    

    def ReadinExp(self, filename):
        df = DataFrame.from_csv(filename, header=None, sep="\t")
        self.Genename = list(df.index)
        TmpExp = np.matrix(copy.deepcopy(df.values))
        for i in range(TmpExp.shape[1]):
            Range = np.max(TmpExp[:,i]) - np.min(TmpExp[:,i])
            TmpExp[:,i] = (TmpExp[:,i] - np.min(TmpExp[:,i])) / Range
        self.ExpMat = TmpExp
        self.ExpMatFix = np.matrix(copy.deepcopy(TmpExp))
        self.NumGene = df.shape[0]
        self.NumExp = df.shape[1]

    def ReadinNet1(self, filename):
        df = DataFrame.from_csv(filename, sep="\t")
        self.TFname = list(df.columns)
        self.Net1 = np.matrix(df.values)
        #self.Net1 = (self.Net1!=0).astype(float)
        #print(np.count_nonzero(self.Net1))
        self.NumTF = df.shape[1]
        Genename = list(df.index)
        if Genename != self.Genename :
            sys.exit("Error: Genes in Net1 do not match Genes in expression!!!")

    def ReadinNet2(self, filename):
        df = DataFrame.from_csv(filename, sep="\t")
        self.Net2 = np.matrix(df.values)
        #self.Net2 = (self.Net2!=0).astype(float)
        TFname = list(df.columns)
        Genename = list(df.index)
        if Genename != self.Genename :
            sys.exit("Error: Genes in Net2 do not match Genes in expression!!!")
        if TFname != self.TFname :
            sys.exit("Error: TFs in Net1 do not match TFs in Net2!!!")

    def ClosedForm4A(self):
        Atmp = np.linalg.inv(self.S.T.dot(self.S)+self.mu*np.eye(self.NumTF)).dot(self.S.T).dot(self.ExpMat)
        self.A = Atmp

    def RegressionWithFixSupport(self):
        for i in range(self.NumGene):
            IndT = np.nonzero(self.S[i,:])
            if IndT[0].size == 0:
                continue
            Afix = self.A[IndT[0], :]
            Sfix = self.ExpMat[i, :].dot(Afix.T).dot(np.linalg.inv(Afix.dot(Afix.T)+self.xi*np.eye(len(Afix))))
            self.S[i, IndT[0]] = Sfix

    def NCA(self):
        for i in range(50):
            self.ClosedForm4A()
            self.RegressionWithFixSupport()
            Error = np.linalg.norm(self.ExpMat - self.S.dot(self.A), 'fro')# + self.xi*np.linalg.norm(self.S, 'fro')**2 + self.mu*np.linalg.norm(self.A, 'fro')**2
        #print(i, Error)
        return Error

    def PriorBoost(self):
        NumEdgeNet1 = np.count_nonzero(self.Net1)
        NumEdgeNet2 = np.count_nonzero(self.Net2)
        NumEdgeT = np.min([NumEdgeNet1, NumEdgeNet2])
        smallnum = np.max([1000, int(NumEdgeT*0.2)])
        PBScore = list()
        EdgeN = list()
        ErrorNet1All = list()
        ErrorNet2All = list()
        for NumE in np.arange(smallnum, int(NumEdgeT*1.15), int(NumEdgeT*0.2)):
            NumEdge = np.min([NumE, NumEdgeT-1])
            ErrorNet1 = 0#np.zeros(1)
            ErrorNet2 = 0#np.zeros(1)
            # Net1Cut
            IndNet1 = np.nonzero(self.Net1)
            ValNet1 = self.Net1[IndNet1[0], IndNet1[1]]
            ValNet1_Sort = np.sort(ValNet1)
            Net1Cut = (self.Net1<ValNet1_Sort[0,NumEdge]).astype(float) - (self.Net1==0).astype(float)
            IndNet1 = np.nonzero(Net1Cut)
            #Net1Mask = np.ones(self.Net1.shape)
            IndNet1U1 = np.unique(IndNet1[0])
            IndNet1U2 = np.unique(IndNet1[1])

            # Net2Cut
            IndNet2 = np.nonzero(self.Net2)
            ValNet2 = self.Net2[IndNet2[0], IndNet2[1]]
            ValNet2_Sort_Ind = np.argsort(ValNet2)
            Net2Cut = np.zeros(Net1Cut.shape)
            Net2Cut[IndNet2[0][ValNet2_Sort_Ind[0,-NumEdge-1:-1]], IndNet2[1][ValNet2_Sort_Ind[0,-NumEdge-1:-1]]] = 1.
            IndNet2 = np.nonzero(Net2Cut)
            #Net2Mask = np.ones(self.Net2.shape)
            IndNet2U1 = np.unique(IndNet2[0])
            IndNet2U2 = np.unique(IndNet2[1])

#print(np.count_nonzero(Net1Cut), np.count_nonzero(Net2Cut))
            
            # overlap between Net1Cut and Net2Cut
            OverlapGene = np.intersect1d(IndNet1U1, IndNet2U1)
            JaccdGene = OverlapGene.size / (IndNet1U1.size + IndNet2U1.size  - OverlapGene.size)
            OverlapTF = np.intersect1d(IndNet1U2, IndNet2U2)
            JaccdTF= OverlapTF.size / (IndNet1U2.size + IndNet2U2.size  - OverlapTF.size)
            #print(OverlapGene.size, IndNet1U1.size, IndNet2U1.size, JaccdGene)
            #print(OverlapTF.size, IndNet1U2.size, IndNet2U2.size, JaccdTF)

            #
            if JaccdTF > 0.5 and JaccdGene > 0.5:
                print("Comparing two networks with top %d edges: " % NumEdge, end="")
                sys.stdout.flush()
                self.S = Net1Cut
                ErrorNet1 = self.NCA()
                self.S = Net2Cut
                ErrorNet2 = self.NCA()
                ErrorNet1All.append(ErrorNet1)
                ErrorNet2All.append(ErrorNet2)
                EdgeN.append(NumEdge)
                print("Fitting Error %f (predict) vs %f (base)" % (ErrorNet1, ErrorNet2))
                sys.stdout.flush()
            else:
                print("Warning:Two compared networks with top %d edges cover different regions of the GRN!!! Not comparable!!!" % NumEdge)
            
            
        self.ErrorNet1All = np.array(ErrorNet1All)
        self.ErrorNet2All = np.array(ErrorNet2All)
        self.EdgeN = EdgeN

    def Plot(self):
        Error1Mean = np.mean(self.ErrorNet1All, axis=1)
        Error2Mean = np.mean(self.ErrorNet2All, axis=1)
        Error1Std = np.std(self.ErrorNet1All, axis=1)
        Error2Std = np.std(self.ErrorNet2All, axis=1)

        print(Error1Mean)
        print(Error2Mean)
        print(Error1Std)
        print(Error2Std)
        print(self.EdgeN)

        p1 = plt.errorbar(self.EdgeN, Error1Mean, yerr=Error1Std)
        p2 = plt.errorbar(self.EdgeN, Error2Mean, yerr=Error2Std)

        plt.ylabel('FitError')
        plt.title('PriorBoost')
        plt.legend((p1[0], p2[0]), ('Net1', 'Net2(Baselie: Expressed-based Method)'))

        plt.show()

    def PriorBoost_Score(self):
        if len(self.ErrorNet1All) < 1:
            print("Cannot compare!!!")
            return 1
        Error1Mean = self.ErrorNet1All#np.mean(self.ErrorNet1All, axis=1)
        Error2Mean = self.ErrorNet2All#np.mean(self.ErrorNet2All, axis=1)
        #Error1Std = np.std(self.ErrorNet1All, axis=1)
        #Error2Std = np.std(self.ErrorNet2All, axis=1)
        print("Cutoffs(same top edges for both networks):", self.EdgeN)
        print("Fitting Error using Predicted Network    :", Error1Mean)
        print("Fitting Error using Baseline Network     :", Error2Mean)

        PBS = Error2Mean - Error1Mean
        PBSMean = np.mean(PBS)
        print("PriorBoost score is %f" % PBSMean)
        if PBSMean > 0:
           print("Predicted Network based on the prior network beat the baseline network!!!")
           print("Indicating the prior network is GOOD!!!")
        else:
            print("Predicted Network based on the prior network cannot beat the baseline network!!!")
            print("Indicating the prior network is BAD!!!")




def main():
    # create parser object
    parser = argparse.ArgumentParser(description = "PriorBoost: Explaination of Predicted Networks Comparing to Baseline Network (Genie3) using Given EXpression!")
    
    # defining arguments for parser object
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-e", type = str, nargs = 1,
                        dest='expfile',
                        required=True,
                        metavar = "expression_file", default = None,
                        help = "<Required> Name of the expression file. How to format: http://")
     
    requiredNamed.add_argument("-p", type = str, nargs = 1,
                        dest='predictedfile',
                        required=True,
                        metavar = "predicted_net_file", default = None,
                        help = "<Required> Name of the predicted network file. How to format: http://")
    
    parser.add_argument( "-b",
                         dest='baselinefile',
                         nargs=1,  # expects ≥ 0 arguments
                         metavar = "baseline_net_file",
                         required=True,
                         default=None,  # default list if no arg value
                         help = "<Required> Name of the baseline network file. How to set: http://"
                        )
    
    # parse the arguments from standard input
    args = parser.parse_args()
    
    if len(sys.argv)==1:
       parser.print_help(sys.stderr)
       sys.exit(1)
    
    # setting parameters
    if args.expfile == None:
        sys.exit("Need expresson data file to run!!!")

    if args.predictedfile == None:
        sys.exit("Need predicted network file to run!!!")

    if args.baselinefile == None:
        sys.exit("Need baseline network file to run!!!")


    PB = PriorBoost(args.expfile[0], args.predictedfile[0], args.baselinefile[0])
    PB.PriorBoost()
    PB.PriorBoost_Score()

    
if __name__ == '__main__':
    main()

