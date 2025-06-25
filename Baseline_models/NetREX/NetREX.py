# -*- coding: utf-8 -*-

import numpy as np
import copy
import scipy.linalg
from pandas import DataFrame
import pandas as pd
import sys
import argparse
import warnings
import progressbar
import os
import torch
import random

def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# hide warnings
warnings.filterwarnings('ignore')
seed_everything(2024)

class NetREX:
    def __init__(self, ExpFile, PriorFile, mu = 1.0, kappa = 1.0, xi = 0.4, IterNum = 1000, Sbound = 1.0, Abound = 1.0):
        # parameters used in NetREX formulation
        self.mu = mu
        self.kappa = kappa
        self.xi = xi
        self.IterNum = IterNum
        self.Sbound = Sbound
        self.Abound = Abound
        self.BootstrapNum = 1
        # read in Exp
        self.ReadinExp(ExpFile)
        # read in Prior
        self.ReadinPrior(PriorFile)
        # construct lap matrix for gene-gene correaltion matrix
        GeneGraph = np.corrcoef(self.ExpMat)
        GeneGraph = (np.absolute(GeneGraph)>0.995)*GeneGraph
        np.fill_diagonal(GeneGraph,0.0)
        Dgraph = np.diag(np.sum(GeneGraph, axis=0))
        self.GeneLapMat = Dgraph - GeneGraph
        # output
        self.S = np.matrix(self.ExistEdge)
        self.SRank = np.zeros(self.S.shape)
        self.A = np.matrix(np.zeros((self.NumTF, self.NumExp)))
        
    def ReadinExp(self, filename):
        df = pd.read_csv(filename, header=None, sep="\t")
        self.Genename = list(df.iloc[:, 0])
        #self.ExpMat = np.matrix(df.values)
        TmpExp = np.matrix(copy.deepcopy(df.values[:, 1:])).astype(float)
        RowSumTmpExp = np.sum(TmpExp, axis=1)
        Ind0 = np.where(RowSumTmpExp==0.)[0]
        if(Ind0.size>0):
            sys.exit("Error: Genes %s's Expression has all zeros!!!" % (self.Genename[Ind0[0]]))
        for i in range(TmpExp.shape[1]):
            Range = np.max(TmpExp[:,i]) - np.min(TmpExp[:,i])
            TmpExp[:,i] = (TmpExp[:,i] - np.min(TmpExp[:,i])) / Range
        self.ExpMat = TmpExp
        self.ExpMatFix = np.matrix(copy.deepcopy(TmpExp))
        self.NumGene = df.shape[0]
        self.NumExp = df.shape[1] - 1

    def ReadinPrior(self, filename):
        df = pd.read_csv(filename, sep="\t")
        self.TFname = list(df.columns[1:])
        self.PriorNet = np.matrix(df.values[:, 1:]).astype(float)
        self.ExistEdge = (df.values[:, 1:]!=0).astype(float)
        self.NumTF = df.shape[1] - 1
        Genename = list(df.iloc[:, 0])
        if Genename != self.Genename :
            sys.exit("Error: Genes in Priro do not match Genes in expression!!!")
        TFtarges = np.sum(self.PriorNet, axis=0)
        if 0 in TFtarges:
            Ind = np.where(TFtarges==0.)
            print("%s has not targets in the prior network!!!" % self.TFname[Ind[1][0]])
            sys.exit("Error: In the prior network, targets of each TF cannot be empty!!!")
        
    
    # Read in user prefered gene-gene network
    def ReadinGeneGeneNet(self, filename):
        df = pd.read_csv(filename, sep="\t")
        RowGeneName = list(df.columns)
        ColGeneName = list(df.index)
        if RowGeneName != ColGeneName:
            sys.exit("The gene-gene network is not symmetric!!!")
        if RowGeneName != self.Genename:
            sys.exit("Error: Genes in gene-gene network do not match Genes in expression!!!")

    def NumEdge_Keep(self, Percentage_Keep):
        NumEdgeinPrior = np.count_nonzero(self.PriorNet)
        self.KeepEdge = round(NumEdgeinPrior * Percentage_Keep)
        
    def NumEdge_Total(self, Total = 1.1):
        NumEdgeinPrior = np.count_nonzero(self.PriorNet)
        self.AddEdge = round(NumEdgeinPrior*Total) - self.KeepEdge
        
    def ClosedForm4A(self):
        Atmp = np.linalg.inv(self.S.T.dot(self.S)+self.mu*np.eye(self.NumTF)).dot(self.S.T).dot(self.ExpMat)
        self.A = np.multiply((np.absolute(Atmp)<=self.Abound), Atmp) + np.multiply((np.absolute(Atmp)>self.Abound), np.sign(Atmp)*self.Abound)
    
    def SolveSylvester(self):
        At = 2*self.kappa*self.GeneLapMat + 2*self.xi*np.eye(self.NumGene)
        Bt = self.A.dot(self.A.T)
        Qt = self.ExpMat.dot(self.A.T)
        Stmp = scipy.linalg.solve_sylvester(At, Bt, Qt)
        self.S = np.multiply((np.absolute(Stmp)<=self.Sbound), Stmp) + np.multiply((np.absolute(Stmp)>self.Sbound), np.sign(Stmp)*self.Sbound)
    
    def Initial_A_S(self):
        self.ClosedForm4A()
        self.SolveSylvester()
        print(self.S)
        print(self.A)
    
    def PALM_S_EdgeControl(self):
        ck = np.linalg.norm(self.A.dot(self.A.T), 'fro') + 2*np.linalg.norm(self.kappa*self.GeneLapMat, 'fro')
        Uk = self.S - (1./ck)*(self.S.dot(self.A).dot(self.A.T) + 2*self.kappa*self.GeneLapMat.dot(self.S) + 2*self.xi*self.S - self.ExpMat.dot(self.A.T))
        UkP = np.multiply((np.absolute(Uk)<=self.Sbound), Uk) + np.multiply((np.absolute(Uk)>self.Sbound), np.sign(Uk)*self.Sbound)
        
        # sepatate existing edges and new edges
        SExist = np.multiply(UkP, self.ExistEdge)
        SExist_ABS = np.absolute(SExist)
        SAdd = np.multiply(UkP, 1-self.ExistEdge)
        SAdd_ABS = np.absolute(SAdd)
        
        # sort and pick
        IdE = np.unravel_index(np.argsort(SExist_ABS.ravel())[0,-int(self.KeepEdge):], SExist_ABS.shape)
        T0 = np.zeros(SExist.shape)
        T0[IdE] = 1.0
        SExist_Pick = np.multiply(T0, SExist)
        IdA = np.unravel_index(np.argsort(SAdd_ABS.ravel())[0,-int(self.AddEdge):], SAdd_ABS.shape)
        T1 = np.zeros(SExist.shape)
        T1[IdA] = 1.0
        SAdd_Pick = np.multiply(T1, SAdd)
        self.S = SExist_Pick + SAdd_Pick
        #print(">")
        #print(self.S)
        #print("<")
        
    def PALM_A(self):
        dk = np.linalg.norm(self.S.dot(self.S.T), 'fro')
        Vk = self.A - (1./dk)*(self.S.T.dot(self.S).dot(self.A) - self.S.T.dot(self.ExpMat))
        Anew = (1./(1+((2*self.mu)/dk)))*Vk
        self.A = np.multiply((np.absolute(Anew)<=self.Abound), Anew) + np.multiply((np.absolute(Anew)>self.Abound), np.sign(Anew)*self.Abound)
        
    def ObjFunction(self):
        Val = 0.5*np.linalg.norm(self.ExpMat - self.S.dot(self.A), 'fro')**2 + self.kappa*np.trace(self.S.T.dot(self.GeneLapMat).dot(self.S)) + self.mu*np.linalg.norm(self.A, 'fro')**2 + self.xi*np.linalg.norm(self.S, 'fro')**2
        return Val
    
    def NetREX_EdgeControl(self):
        self.ClosedForm4A()
        self.SolveSylvester()
        if self.NumGene*self.NumTF > 100000:
            Estop = 1
        else:
            Estop = 1e-10
        #print(self.S)
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                    progressbar.Percentage(), ' ',
                                                    progressbar.ETA()])
        Valold = self.ObjFunction()
        for i in progress(range(self.IterNum)):
            self.PALM_S_EdgeControl()
            self.PALM_A()
            Valnew = self.ObjFunction()
            #print(i,Valnew, abs(Valold-Valnew))
            if(abs(Valold-Valnew) < Estop):
                print("Converge!")
                sys.stdout.flush()
                break
            Valold = Valnew
        #print(self.S)
        
    def RankEdge(self):
        self.SRank = np.zeros(self.S.shape)
        EdgeInd = np.nonzero(self.S)
        NumN0 = np.count_nonzero(self.S)
        RankGroup = np.zeros((20,NumN0))
        #print(RankGroup)
        for ii in range(20):
            RandSampleId = np.random.random_integers(self.NumExp,size=(self.NumExp))-1
            #print(len(set(RandSampleId)))
            ExpSampled = self.ExpMat[:,RandSampleId]
            TFASampled = self.A[:,RandSampleId]
            Conf = np.zeros(NumN0)
            for i in range(NumN0):
                IndGene = EdgeInd[0][i]
                IndTF = EdgeInd[1][i]
                Par = copy.deepcopy(self.S[IndGene,:])
                #print(Par)
                Sigma_full = np.linalg.norm(ExpSampled[IndGene,:] - Par.dot(TFASampled), 'fro')**2
                Par[0,IndTF] = 0.
                #print(Par)
                Sigma_miss = np.linalg.norm(ExpSampled[IndGene,:] - Par.dot(TFASampled), 'fro')**2
                Conf[i] = 1. - (Sigma_full / Sigma_miss)
            #print(Conf)
            RankId = np.argsort(-Conf)
            ConfRank = np.zeros(NumN0)
            ConfRank[RankId] = np.arange(NumN0) + 1.
            #print(ConfRank)
            RankGroup[ii,:] = ConfRank
        #print(RankGroup)
        RankGroupMean = np.mean(RankGroup, axis=0)
        RankIdt = np.argsort(RankGroupMean)
        FinalRank = np.zeros(NumN0)
        FinalRank[RankIdt] = np.arange(NumN0) + 1.
        self.SRank[EdgeInd[0], EdgeInd[1]] = FinalRank        

    def NetREX_BootStrap(self):
        self.SRankFinal = np.zeros(self.S.shape)
        SRankF = np.zeros(self.S.shape)
        RankGroup = np.zeros((self.BootstrapNum,1), dtype=np.ndarray)
        for i in range(self.BootstrapNum):
            print("\tBootstrap %d ... " % (i+1))
            sys.stdout.flush()
            self.S = np.matrix(self.ExistEdge)
            self.SRank = np.zeros(self.S.shape)
            self.A = np.matrix(np.zeros((self.NumTF, self.NumExp)))
            RandSampleId = np.random.random_integers(self.NumExp,size=(self.NumExp))-1
            self.ExpMat = self.ExpMatFix[:,RandSampleId]
            self.NetREX_EdgeControl()
            self.RankEdge()
            RankGroup[i][0] = copy.deepcopy(self.SRank)
            SRankF = SRankF + copy.deepcopy(self.SRank)
            #print("Done!")
        
        # rank edges 
        SRankFMask = (SRankF>0).astype(float)
        SRankFinal = np.zeros(self.S.shape)
        SRankTimes = np.zeros(RankGroup[0][0].shape)
        for i in range(self.BootstrapNum):
            Tmp1 = copy.deepcopy(RankGroup[i][0])
            Tmp1Mask = (Tmp1!=0).astype(float)
            Tmp2 = np.zeros(Tmp1.shape)
            Tmp2 = SRankFMask - Tmp1Mask
            Rank0 = np.count_nonzero(Tmp2)*np.count_nonzero(Tmp1) + np.sum(np.arange(1,np.count_nonzero(Tmp2)+1))
            if np.count_nonzero(Tmp2) == 0:
                Rank0 = float(Rank0)
            else:
                Rank0 = float(Rank0) / np.count_nonzero(Tmp2)
            SRankFinal = SRankFinal + Tmp1 + Tmp2*Rank0
            SRankTimes = SRankTimes + Tmp1Mask

        # consider frequence
        Indn0 = np.nonzero(SRankTimes)
        Rankings = SRankFinal[Indn0[0], Indn0[1]]
        Times = SRankTimes[Indn0[0], Indn0[1]]
        Scores = np.multiply(Rankings, 1./Times)
        SRankFinal[Indn0[0], Indn0[1]] = Scores
        
        # Final Rank
        Ind = np.nonzero(SRankFinal)
        Val = SRankFinal[Ind[0], Ind[1]]
        RankIdt = np.argsort(Val)
        NumN0 = np.count_nonzero(SRankFinal)
        Val[RankIdt] = np.arange(NumN0) + 1.
        self.SRankFinal[Ind[0], Ind[1]] = Val

        
def CominbRanks(RankGroup):
    NumRank = RankGroup.size
    SRankF = np.zeros(RankGroup[0][0].shape)
    for i in range(NumRank):
        SRankF = SRankF + RankGroup[i][0]
    
    SRankFMask = (SRankF>0).astype(float)
    SRankFinal = np.zeros(RankGroup[0][0].shape)
    SRankTimes = np.zeros(RankGroup[0][0].shape)
    for i in range(NumRank):
        Tmp1 = copy.deepcopy(RankGroup[i][0])
        Tmp1Mask = (Tmp1!=0).astype(float)
        Tmp2 = np.zeros(Tmp1.shape)
        Tmp2 = SRankFMask - Tmp1Mask
        Rank0 = np.count_nonzero(Tmp2)*np.count_nonzero(Tmp1) + np.sum(np.arange(1,np.count_nonzero(Tmp2)+1))
        if np.count_nonzero(Tmp2) == 0:
            Rank0 = float(Rank0)
        else:
            Rank0 = float(Rank0) / np.count_nonzero(Tmp2)
        SRankFinal = SRankFinal + Tmp1 + Tmp2*Rank0
        SRankTimes = SRankTimes + Tmp1Mask
        #print(Tmp1 + Tmp2*Rank0)
    
    # consider frequence
    Indn0 = np.nonzero(SRankFMask)
    Rankings = SRankFinal[Indn0[0], Indn0[1]]
    Times = SRankTimes[Indn0[0], Indn0[1]]
    Scores = np.multiply(Rankings, 1./Times)
    SRankFinal[Indn0[0], Indn0[1]] = Scores
    
    # Final Rank
    Ind = np.nonzero(SRankFinal)
    Val = SRankFinal[Ind[0], Ind[1]]
    RankIdt = np.argsort(Val)
    NumN0 = np.count_nonzero(SRankFinal)
    Val[RankIdt] = np.arange(NumN0) + 1.
    SRankFinal[Ind[0], Ind[1]] = Val
    
    return SRankFinal

def OutputGRN(RankNet, GeneName, TFName, output_dir):
    NetAdjFile = os.path.join(output_dir, "NetREX_PredictedNetwork.tsv")
    NAfile = open(NetAdjFile, 'w')
    NAfile.write("\t")
    for i in range(len(TFName)-1):
        NAfile.write("%s\t" % TFName[i])
    NAfile.write("%s\n" % TFName[-1])
    
    for i in range(len(GeneName)):
        NAfile.write("%s\t" % GeneName[i])
        for j in range(len(TFName)-1):
            NAfile.write("%f\t" % RankNet[i,j])
        NAfile.write("%f\n" % RankNet[i,-1])
    NAfile.close()
    
    EdgeListFile = os.path.join(output_dir, "NetREX_PredictedEdgeList.txt")
    ELfile = open(EdgeListFile, 'w')
    Ind = np.nonzero(RankNet)
    Val = RankNet[Ind[0], Ind[1]]
    RankId = np.argsort(Val)
    ELfile.write("TF\tGene\tRank\n")
    for i in range(len(Val)):
        ELfile.write("%s\t%s\t%f\n" % (TFName[Ind[1][RankId[i]]], GeneName[Ind[0][RankId[i]]], Val[RankId[i]]) )
    ELfile.close()


def main():
    # create parser object
    parser = argparse.ArgumentParser(description = "NetREX: Network Rewiring using EXpression!")
    
    # defining arguments for parser object
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-e", type = str, nargs = 1,
                        dest='expfile',
                        required=True,
                        metavar = "expression_file", default = None,
                        help = "<Required> Name of the expression file. How to format: http://")
     
    requiredNamed.add_argument("-p", type = str, nargs = 1,
                        dest='priorfile',
                        required=True,
                        metavar = "prior_file", default = None,
                        help = "<Required> Name of the prior network file. How to format: http://")
    
    requiredNamed.add_argument("-o", type = str, nargs = 1,
                        dest='outputdir',
                        required=True,
                        metavar = "output_dir", default = None,)
    
    parser.add_argument( "-k",
                         dest='keepedge',
                         nargs="+",  # expects ≥ 0 arguments
                         metavar = "a_list",
                         default=None,  # default list if no arg value
                         help = "Control # edges to be kept in the prior. How to set: http://"
                        )
    
    parser.add_argument( "-t",
                         dest='totaledge', 
                         nargs=1,  # expects ≥ 0 arguments
                         metavar = "a_ratio",
                         type=None,
                         default=None,  # 1.1*number of edges in the prior network
                         help = "Control # edges in the final prediction. How to set: http:// ",
                       )
    
    # parse the arguments from standard input
    args = parser.parse_args()
    
    if len(sys.argv)==1:
       parser.print_help(sys.stderr)
       sys.exit(1)
    
    # setting parameters
    if args.expfile == None:
        sys.exit("Need expresson data file to run!!!")

    if args.priorfile == None:
        sys.exit("Need prior network file to run!!!")
    
    #print(args.expfile, args.priorfile )
    #print(args.keepedge)
    
    NetREX_tmp = NetREX(args.expfile[0], args.priorfile[0])

    # automactially set -k -t
    EdgeDensity = np.count_nonzero(NetREX_tmp.PriorNet) / (float(NetREX_tmp.NumGene)*NetREX_tmp.NumTF)
    if args.keepedge == None:
        if EdgeDensity > 0.5:
            NumEdge_Keep = np.arange(0.6,0.8,0.1)
        else:
            NumEdge_Keep = np.arange(0.6,0.9,0.1)
    else:
        NumEdge_Keep = args.keepedge
    
    if args.totaledge == None:
        if EdgeDensity > 0.5:
            NumEdge_Total = [1.0]
        elif EdgeDensity > 0.05:
            NumEdge_Total = [1.2]
        elif EdgeDensity > 0.01:
            NumEdge_Total = [1.8]
    else:
        NumEdge_Total = args.totaledge
#NumEdge_Total = float(NumEdge_T[0])
    
    del NetREX_tmp

    print(NumEdge_Keep)

    # run NetREX
    RankGroup = np.zeros((len(NumEdge_Keep),1), dtype=np.ndarray)
    for i in range(len(NumEdge_Keep)):
        NetREX_Example = NetREX(args.expfile[0], args.priorfile[0])
        NetREX_Example.NumEdge_Keep(float(NumEdge_Keep[i]))
        NetREX_Example.NumEdge_Total(float(NumEdge_Total[0]))
        print("Setting %d: Keep %d edges in prior and add %d edges." % (i+1, int(NetREX_Example.KeepEdge), int(NetREX_Example.AddEdge)))
        NetREX_Example.NetREX_BootStrap()
        RankGroup[i][0] = copy.deepcopy(NetREX_Example.SRankFinal)

    print("Consensus all settings ... ", end="")
    FinalRankNet = CominbRanks(RankGroup)
    print("Done!")
    print("Writing the predicted network to files ... ", end="")
    OutputGRN(FinalRankNet, NetREX_Example.Genename, NetREX_Example.TFname, args.outputdir[0])
    print("Done!")
    
if __name__ == '__main__':
    main()
