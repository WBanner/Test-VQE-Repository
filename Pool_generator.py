#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:59:39 2020

@author: Vlad 
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

"""


import itertools
import numpy as np
from random import shuffle


#Global variable, defining the number of qubits ()
QubitNumber =4



#For a list of Pauli strings(Pool), swaps qubits i and j
def Swap(Pool,i,j):
    NewPool=[]
    for element in Pool:
        newElement=''
        for k in range(len(element)):
            if k==i:
                newElement+=element[j]
            elif k==j:
                newElement+=element[i]
            else:
                newElement+=element[k]
        NewPool.append(newElement)
    return NewPool
        


#Tests if a list of Pauli strings(Pool) forms a complete pool
def TestPool(Pool):
    ControlAlgebra=PauliAlgebra(Pool, Group=True, Algebra=True)
    if ControlAlgebra.CheckPool():
        print('The pool is complete')
        return True
    else:
        print('The pool is incomplete')
        return False



#Tests if a list of Pauli strings(Pool) can be split into two groups, commuting with each other
def CheckInseparability(Pool):
    Set=[Pool[0]]
    i=0
    while i<len(Set):
        for PoolElement in Pool:
            if PoolElement not in Set:
                if not CheckCommutivity(PoolElement, Set[i]):
                    Set.append(PoolElement)
        i+=1
    if len(Set)<len(Pool):
        return False
    else:
        return True
    


       


#Constructs a recursive complete pool (the one I did the completeness proof for)
def CompletePool():
    CompleteGenerators=[]
    for i in range(QubitNumber):
        if i==0:
            CompleteGenerators.append('Y')
        elif i==1:
            CompleteGenerators[0]='Z'+CompleteGenerators[0]
            CompleteGenerators.append('YI')
        else:
            for j in range(len(CompleteGenerators)):
                CompleteGenerators[j]='Z'+CompleteGenerators[j]
            CompleteGenerators.append('Y'+'I'*i)
            CompleteGenerators.append('I'+'Y'+'I'*(i-1))
    return CompleteGenerators


#Returns the nuber of consecutive qubits, connected in the Pauli string a (IXZIYII has connectivity 4)
def CheckConnectivity(a):
    Connectivity=0
    i=0
    while i<len(a) and a[i]=='I' :
        i+=1
    if i==len(a):
        return Connectivity
    else:
        Connectivity=1
    for j in range(i+1,len(a)):
        if a[j]!='I':
            Connectivity=j-i+1
    return Connectivity
        

#Constructs all odd strings of given connectivity n
def OddConnectivityStrings(n):
    Set=['I','Y','X','Z']
    MyList=['']
    TmpList=[]
    i=0
    while i<QubitNumber:
        for element in MyList:
            for k in range(4):
                tmp=element+Set[k]
                if CheckConnectivity(tmp)<=n:
                    TmpList.append(tmp)
        i+=1
        MyList=TmpList.copy()
        TmpList=[]
    for element in MyList:
        if CheckIfOdd(element):
            TmpList.append(element)
    MyList=TmpList
    return MyList
            
    



#The function checks if the strings a and b commute and returns True in that case (False otherwise)
def CheckCommutivity(a,b):
    result=1
    for i in range(QubitNumber):
        if  (a[i]=='X' and b[i]=='Y') or (a[i]=='X' and b[i]=='Z') or\
            (a[i]=='Y' and b[i]=='X') or (a[i]=='Y' and b[i]=='Z') or\
            (a[i]=='Z' and b[i]=='X') or (a[i]=='Z' and b[i]=='Y'):
            result*=-1
    if result==1:
        return True
    else:
        return False

#The function checks if strings a is odd and returns True in that case (False otherwise)
def CheckIfOdd(a):
    result=-1
    for i in range(QubitNumber):
        if (a[i]=='Y'):
            result*=-1
    if result==1:
        return True
    else:
        return False


#The function multiplies strings a and b and returns their product
def MultiplyStrings(a,b):
    result=''
    for i in range(QubitNumber):
        if a[i]=='X' and b[i]=='Y':
            result+='Z'
            
        if a[i]=='X' and b[i]=='Z':
            result+='Y'
		
        if a[i]=='Y' and b[i]=='X':
            result+='Z'
			
        if a[i]=='Y' and b[i]=='Z':
            result+='X'
			
        if a[i]=='Z' and b[i]=='X':
            result+='Y'
		
        if a[i]=='Z' and b[i]=='Y':
            result+='X'
			
        if a[i]=='Z' and b[i]=='Z':
            result+='I'
			
        if a[i]=='X' and b[i]=='X':
            result+='I'
			
        if a[i]=='Y' and b[i]=='Y':
            result+='I'
			
        if a[i]=='I':
            result+=b[i]
			
        if b[i]=='I' and a[i]!='I':
            result+=a[i]

    return result


    


	

        
       
#The objects of this class is built from a pool and contains 1)the Product Pauli group, 
#generated by the pool, 2) the subgroup of elements, containing I and Z only
#3) The algebra, generated by the pool. One can define if the algebras ans groups should be computed
#(change the default parameters Algebra and Group to False not to construct them)        
class PauliAlgebra:

#The function to input the generator pool  or add elements to the existing pool
    def AddStrings(self):
        tmp = input('Insert Pauli Strings of length '+str(QubitNumber)+'or type exit to finish\n')
        while tmp!='exit':
            if tmp not in self.PauliStrings:
                self.Counter+=1
                self.PauliStrings.append(tmp)
            if tmp not in self.ProductStrings:
                self.ProductCounter+=1
                self.ProductStrings.append(tmp)
                if 'X' not in tmp and 'Y' not in tmp:
                    self.ZsubgroupCounter+=1
                    self.Zsubgroup.append(tmp)
            tmp = input()
        self.PauliStrings.sort()
        self.ProductStrings.sort()
        
#Constructs a class object from an input pool provided as a parameter or from keyboard        
    def __init__(self,InputGenerators=None,Group=None,Algebra=None):
        if Group==None:
            Group=True
        if Algebra==None:
            Algebra=True
        if InputGenerators==None:
            self.PauliStrings=[]
            self.ProductStrings=[]
            self.Zsubgroup=[]
            self.ProductCounter=0
            self.ZsubgroupCounter=0
            self.Counter=0
            self.AddStrings()
        else:
            self.PauliStrings=InputGenerators.copy()
            self.ProductStrings=InputGenerators.copy()
            self.Counter=len(InputGenerators)
            self.ProductCounter=len(InputGenerators)
            self.Zsubgroup=[]
            self.ZsubgroupCounter=0
            for element in InputGenerators:
                if 'X' not in element and 'Y' not in element:
                    self.ZsubgroupCounter+=1
                    self.Zsubgroup.append(element)
        if Algebra==True:
            self.GeneratePauliStrings()
        if Group==True:
            self.GenerateGroup()
        self.PauliStrings.sort()
        self.ProductStrings.sort()


#Generates an algebra from a pool in self.PauliStrings and saves it there  
### Takes extremely long to compute for 8 qubits and more        
    def GeneratePauliStrings (self):
        MyList=self.PauliStrings.copy()
        tmpList=self.PauliStrings.copy()
        while len(tmpList)!=0:
            length=len(tmpList)
            for i in range(length):
                for j in range(len(MyList)):
                    if not CheckCommutivity(MyList[j], tmpList[i]):
                        tmp=MultiplyStrings(tmpList[i],MyList[j])
                        if tmp not in MyList and tmp not in tmpList:
                            tmpList.append(tmp)
            tmpList=tmpList[length:]
            MyList=MyList+tmpList
        self.Counter=len(MyList)
        self.PauliStrings=MyList


    
    
    
#Prints all strings in self.PauliStrings 
    def printAlgebra(self,start=0,Str=None):
        if Str==None:
            Str=[]
            for i in range(QubitNumber):
                Str.append('0')
            
        for i in range(start,QubitNumber):
            Str[i]='1'
            for r in range(len(Str)):
                print(Str[r],end='')
            print(':',end=' ')
            first=0
            for k in range(self.Counter):
                flag=0
                for j in range(QubitNumber):
                    if (self.PauliStrings[k][j]=='Y' and Str[j]!='1'):
                        flag=1
                    if (self.PauliStrings[k][j]=='X' and Str[j]!='1'):
                        flag=1
                    if (self.PauliStrings[k][j]=='I' and Str[j]=='1'):
                        flag=1
                    if (self.PauliStrings[k][j]=='Z' and Str[j]=='1'):
                        flag=1
                if flag==0:
                    if first==0:
                        print(self.PauliStrings[k],end='')
                        first=1
                    else:
                        print(',',self.PauliStrings[k],end='')
            print('')
            Str[i]='0'
	
	
        for i in range(start,QubitNumber):
            Str[i]='1'
            self.printAlgebra(i+1,Str)
            Str[i]='0'
        if start==0:
            print()


#Prints a product group from the self.ProductStrings	    
    def printGroup(self,start=0,Str=None):
        if Str==None:
            Str=[]
            for i in range(QubitNumber):
                Str.append('0')
            for r in range(len(Str)):
                print(Str[r],end='')
            print(':',end=' ')
            first=0
            for k in range(self.ProductCounter):
                flag=0
                for j in range(QubitNumber):
                    if (self.ProductStrings[k][j]=='Y' and Str[j]!='1'):
                        flag=1
                    if (self.ProductStrings[k][j]=='X' and Str[j]!='1'):
                        flag=1
                    if (self.ProductStrings[k][j]=='I' and Str[j]=='1'):
                        flag=1
                    if (self.ProductStrings[k][j]=='Z' and Str[j]=='1'):
                        flag=1
                if flag==0:
                    if first==0:
                        print(self.ProductStrings[k],end='')
                        first=1
                    else:
                        print(',',self.ProductStrings[k],end='')
            print('')
            
        for i in range(start,QubitNumber):
            Str[i]='1'
            for r in range(len(Str)):
                print(Str[r],end='')
            print(':',end=' ')
            first=0
            for k in range(self.ProductCounter):
                flag=0
                for j in range(QubitNumber):
                    if (self.ProductStrings[k][j]=='Y' and Str[j]!='1'):
                        flag=1
                    if (self.ProductStrings[k][j]=='X' and Str[j]!='1'):
                        flag=1
                    if (self.ProductStrings[k][j]=='I' and Str[j]=='1'):
                        flag=1
                    if (self.ProductStrings[k][j]=='Z' and Str[j]=='1'):
                        flag=1
                if flag==0:
                    if first==0:
                        print(self.ProductStrings[k],end='')
                        first=1
                    else:
                        print(',',self.ProductStrings[k],end='')
            print('')
            Str[i]='0'
	
	
        for i in range(start,QubitNumber):
            Str[i]='1'
            self.printGroup(i+1,Str)
            Str[i]='0'
        if start==0 :
            print()
	    

#Returns a subset of PauliStrings, containig the biggest possible number of mutually anticommuting strings
    def BiggestAntiCommutingSet(self,AntiCommutingSet=None,MaxAntiCommutingSet=None):
        if AntiCommutingSet is None:
            AntiCommutingSet=[]
        if MaxAntiCommutingSet is None:
            MaxAntiCommutingSet=[]    
        for i in range(self.Counter):
            if (len(AntiCommutingSet)==0):
                AntiCommutingSet.append(self.PauliStrings[i])
                MaxAntiCommutingSet=self.BiggestAntiCommutingSet(AntiCommutingSet,MaxAntiCommutingSet)
                if len(AntiCommutingSet)>len(MaxAntiCommutingSet):
                    MaxAntiCommutingSet=AntiCommutingSet.copy()
                AntiCommutingSet.pop(-1)
            else:
                flag=0
                for j in range(len(AntiCommutingSet)):
                    if CheckCommutivity(self.PauliStrings[i],AntiCommutingSet[j]):
                        flag=1
                if (flag==0):
                    AntiCommutingSet.append(self.PauliStrings[i])
                    MaxAntiCommutingSet=self.BiggestAntiCommutingSet(AntiCommutingSet,MaxAntiCommutingSet)
                    if len(AntiCommutingSet)>len(MaxAntiCommutingSet):
                        for i in range(len(AntiCommutingSet)):
                            MaxAntiCommutingSet=AntiCommutingSet.copy()
                    AntiCommutingSet.pop(-1)
        return MaxAntiCommutingSet 
    




#Computes the nunmber of strings in PauliStrings, anticommuting with the string a
    def HowManyAntiCommute(self, a):
        num=0
        for i in range(self.Counter):
            if not CheckCommutivity(self.PauliStrings[i],a):
                num+=1
        return num
       


 

    
#Generates a product group from a pool in self.ProductStrings and saves it there     
    def GenerateGroup (self):
        self.ZsubgroupCounter=1
        tmpList=['I'* QubitNumber]
        self.Zsubgroup.append(tmpList[0])
        for i in range(len(self.ProductStrings)):
            if self.ProductStrings[i] not in tmpList:
                for j in range(len(tmpList)):
                    tmp=MultiplyStrings(self.ProductStrings[i],tmpList[j])
                    tmpList.append(tmp)
                    if 'X' not in tmp and 'Y' not in tmp:
                        self.ZsubgroupCounter+=1
                        self.Zsubgroup.append(tmp)
        self.ProductCounter=len(tmpList)
        self.ProductStrings=tmpList
        
                    
#performs a similarity transformation with operator 'Operator'
#on the strings in self.ProductString,self.PauliStrings,self.Zsubgroup                   
    def SimilarityTransformation(self, Operator):
        if not CheckIfOdd(Operator):
            return None
        for i in range(len(self.ProductStrings)):
            if not CheckCommutivity(Operator, self.ProductStrings[i]):
                self.ProductStrings[i]=MultiplyStrings(Operator, self.ProductStrings[i])
        self.Zsubgroup=[]
        for element in self.ProductStrings:
            if 'X' not in element and 'Y' not in element:
                self.Zsubgroup.append(element)
        for i in range(len(self.PauliStrings)):
            if not CheckCommutivity(Operator, self.PauliStrings[i]):
                self.PauliStrings[i]=MultiplyStrings(Operator, self.PauliStrings[i])
        self.PauliStrings.sort()
        self.ProductStrings.sort()
        

#returns a subset of ProductStrings, with connectivity not bigger than n
    def ConnectivitySubSet(self,n):
        connectivitysubset=[]
        for element in self.ProductStrings:
            if CheckConnectivity(element)<=n and CheckIfOdd(element) :
                connectivitysubset.append(element)
        return connectivitysubset
        


#Checks if the object self was generated from a complete pool of operators
    def CheckPool(self):
        if self.CheckGroup() and self.CheckAlgebraSize():
            return True
        else:
            return False
        
    def CheckGroup(self):
        if self.ZsubgroupCounter==2**(QubitNumber-2) and\
            self.ProductCounter==2**(2*QubitNumber-2):
                Str=''
                my_array=np.zeros((2**QubitNumber-1,), dtype=int)
                for element in self.ProductStrings:
                    if CheckIfOdd(element):
                        for letter in element:
                            if letter=='I' or letter=='Z':
                                Str=Str+'0'
                            if letter=='Y' or letter=='X':
                                Str=Str+'1'
                        my_array[int(Str,2)-1]=1
                        Str=''
                if 0 not in my_array:
                    return True
                else:
                    return False
        else:
            return False
  
#checks if the algebra size in self.PauliStrings is 2**(QubitNumber-2)*(2**(QubitNumber-1)+1)                
    def CheckAlgebraSize(self):
        if len(self.PauliStrings)==2**(QubitNumber-2)*(2**(QubitNumber-1)+1):
            return True
        else:
            return False
        
        
#checks if the strings in in self.PauliStrings can flip any subset of qubits (necessary condition of completeness)
    def CheckFlippings(self,start=0):
        Str=''
        my_array=np.zeros((2**QubitNumber-1,), dtype=int)
        for element in self.PauliStrings:
            for letter in element:
                if letter=='I' or letter=='Z':
                    Str=Str+'0'
                if letter=='Y' or letter=='X':
                    Str=Str+'1'
            my_array[int(Str,2)-1]=1
            Str=''
        if 0 not in my_array:
            return True
        else:
            return False



#####################
            
#THE FUNCTIONS BELOW CONSTRUCT THE COMPLETE POOLS. UNCOMMENT THE OUTPUT PART TO PRINT THEM DURING THE RUN 

#Conjecture based functions create the pools, that were checked to be complete for small number of qubits
            
#To make sure the conjecture is true, use the function TestPool(Pool)  
            
#####################

#Finds all possible complete pools of connectivity n, based on inseparability conjecture 
def ConjectureBasedCompletePools(n):
    CompletePools=[]
    CompleteCounter=0         
    Set=OddConnectivityStrings(n) 
    shuffle(Set)
    
    for element in itertools.combinations(Set, 2*QubitNumber-2):
        if CheckInseparability(list(element)):
            tmp=PauliAlgebra(list(element),Group=True,Algebra=False)
            if tmp.CheckGroup():
                CompleteCounter+=1
                CompletePools.append(list(element))
                if CompleteCounter % 1 ==0:
                    for element in CompletePools[CompleteCounter-1:CompleteCounter]:
                       print(element)
    return CompletePools

#Finds all possible complete pools of connectivity n, based on the algebra size    
def ProvenCompletePools(n):
    CompletePools=[]
    CompleteCounter=0         
    Set=OddConnectivityStrings(n) 
    shuffle(Set)
    
    for element in itertools.combinations(Set, 2*QubitNumber-2):
        if CheckInseparability(list(element)):
            tmp=PauliAlgebra(list(element),Group=True,Algebra=False)
            if tmp.CheckGroup():
                tmp.GeneratePauliStrings()
                if tmp.CheckAlgebraSize():
                    CompleteCounter+=1
                    CompletePools.append(list(element))
#                    if CompleteCounter % 1 ==0:
#                        for element in CompletePools[CompleteCounter-1:CompleteCounter]:
#                            print(element)  
    return CompletePools
  
    

#Finds the complete pools of connectivity n, that generate a specific group(similarity transformations not included)
#based on inseparability conjecture. A similarity transformation can be included if necessary  
def ConjectureBasedSubset(n,Operator=None):
    
    if Operator==None:
        Operator='I'*QubitNumber
        
    Group=PauliAlgebra(CompletePool(),Algebra=False)
    Group.SimilarityTransformation(Operator)
    Set=Group.ConnectivitySubSet(n)
    shuffle(Set)
    
    CompletePools=[]
    CompleteCounter=0
    
    for element in itertools.combinations(Set, 2*QubitNumber-2):
        if CheckInseparability(list(element)):
            tmp=PauliAlgebra(list(element),Group=True,Algebra=False)
            if tmp.CheckGroup():
                CompleteCounter+=1
                CompletePools.append(list(element))
#                if CompleteCounter % 1 ==0:
#                    for element in CompletePools[CompleteCounter-1:CompleteCounter]:
#                       print(element)
    return CompletePools







#Finds the complete pools of connectivity n, that generate a specific group(similarity transformations not included)
#based on the algebra size. A similarity transformation can be included if necessary     
def ProvenSubset(n,Operator=None):
    if Operator==None:
        Operator='I'*QubitNumber
        
    Group=PauliAlgebra(CompletePool(),Algebra=False)
    Group.SimilarityTransformation(Operator)
    Set=Group.ConnectivitySubSet(n)
    shuffle(Set)
    
    CompletePools=[]
    CompleteCounter=0 
    
    for element in itertools.combinations(Set, 2*QubitNumber-2):
        if CheckInseparability(list(element)):
            tmp=PauliAlgebra(list(element),Group=True,Algebra=False)
            if tmp.CheckGroup():
                tmp.GeneratePauliStrings()
                if tmp.CheckAlgebraSize():
                    CompleteCounter+=1
                    CompletePools.append(list(element))
#                    if CompleteCounter % 1 ==0:
#                        for element in CompletePools[CompleteCounter-1:CompleteCounter]:
#                            print(element)  
    return CompletePools




#print('0')
#ConjectureBasedCompletePools(0)
#print('1')
#ConjectureBasedCompletePools(1)
#print('2')
#ConjectureBasedCompletePools(2)
#ConjectureBasedSubset(3,'IXIZ')
#print('3')
#ConjectureBasedCompletePools(3)
#print(len(ConjectureBasedSubset(2)))
#print(len(ProvenSubset(2)))
#print(len(ConjectureBasedCompletePools(3)))
#print(len(ProvenCompletePools(2)))

#TestPool( Insert Pool list )