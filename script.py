import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
from scipy import linalg


import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD

    a=np.hsplit(X,2)
    col1=a[0]
    col2=a[1]
    comp=np.array([1,2,3,4,5])
    sums = np.zeros(shape=(5,))
    counts = np.zeros(shape=(5,))

    for i in range(0,150):
        if y[i] == comp[0]:
            sums[0]=sums[0]+col1[i]
            counts[0] = counts[0] +1
        elif y[i] == comp[1]:
            sums[1]=sums[1]+col1[i]
            counts[1] = counts[1] +1
        elif y[i]==comp[2]:
            sums[2]=sums[2]+col1[i]
            counts[2] = counts[2] +1
        elif y[i]==comp[3]:
            sums[3]=sums[3]+col1[i]
            counts[3] = counts[3] +1
        else:
            sums[4]=sums[4]+col1[i]
            counts[4] = counts[4] +1


    sums1=np.zeros(shape=(5,))
    counts1 = np.zeros(shape=(5,))

    for i in range(0,150):
        if y[i] == comp[0]:
            sums1[0]=sums1[0]+col2[i]
            counts1[0] = counts1[0] +1
        elif y[i]==comp[1]:
            sums1[1]=sums1[1]+col2[i]
            counts1[1] = counts1[1] +1
        elif y[i]==comp[2]:
            sums1[2]=sums1[2]+col2[i]
            counts1[2] = counts1[2] +1
        elif y[i]==comp[3]:
            sums1[3]=sums1[3]+col2[i]
            counts1[3] = counts1[3] +1
        else:
            sums1[4]=sums1[4]+col2[i]
            counts1[4] = counts1[4] +1


    mu1 = np.divide(sums,counts)
    mu2=np.divide(sums1,counts1)
    means = np.vstack((mu1,mu2))
    csum = X.sum(axis=0)
    mean = csum/150
    var=np.subtract(col1,mean[0])
    sq=np.square(var)
    summed=sq.sum(axis=0)
    cov1=summed/149
    var1=np.subtract(col2,mean[1])
    sq1=np.square(var1)
    summed1=sq1.sum(axis=0)
    cov2=summed1/149
    mvar=np.multiply(var,var1)
    ad=mvar.sum(axis=0)
    cov3=ad/149
    fcov=np.array([cov1,cov3],dtype=float)
    scov=np.array([cov3,cov2],dtype=float)
    covmat=np.hstack((fcov,scov))
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD


    a=np.hsplit(X,2)
    col1=a[0]
    col2=a[1]
    comp=np.array([1,2,3,4,5])
    sums = np.zeros(shape=(5,))
    counts = np.zeros(shape=(5,))

    for i in range(0,150):
        if y[i] == comp[0]:
            sums[0]=sums[0]+col1[i]
            counts[0] = counts[0] +1
        elif y[i] == comp[1]:
            sums[1]=sums[1]+col1[i]
            counts[1] = counts[1] +1
        elif y[i]==comp[2]:
            sums[2]=sums[2]+col1[i]
            counts[2] = counts[2] +1
        elif y[i]==comp[3]:
            sums[3]=sums[3]+col1[i]
            counts[3] = counts[3] +1
        else:
            sums[4]=sums[4]+col1[i]
            counts[4] = counts[4] +1

    sums1=np.zeros(shape=(5,))
    counts1 = np.zeros(shape=(5,))

    for i in range(0,150):
        if y[i] == comp[0]:
            sums1[0]=sums1[0]+col2[i]
            counts1[0] = counts1[0] +1
        elif y[i]==comp[1]:
            sums1[1]=sums1[1]+col2[i]
            counts1[1] = counts1[1] +1
        elif y[i]==comp[2]:
            sums1[2]=sums1[2]+col2[i]
            counts1[2] = counts1[2] +1
        elif y[i]==comp[3]:
            sums1[3]=sums1[3]+col2[i]
            counts1[3] = counts1[3] +1
        else:
            sums1[4]=sums1[4]+col2[i]
            counts1[4] = counts1[4] +1


    mu1 = np.divide(sums,counts)
    mu2=np.divide(sums1,counts1)
    means = np.vstack((mu1,mu2))


    a=np.hsplit(means,5)
    mu1=a[0]
    mu2=a[1]
    mu3=a[2]
    mu4=a[3]
    mu5=a[4]

    #print counts


    Xsubs1=np.empty(shape=(0,2))

    Xsubs2=np.empty(shape=(0,2))

    Xsubs3=np.empty(shape=(0,2))

    Xsubs4=np.empty(shape=(0,2))

    Xsubs5=np.empty(shape=(0,2))

    for i in range(0,150):
        if y[i] == comp[0]:
            temp1=X[i,:]-np.transpose(mu1)
            Xsubs1 = np.append(Xsubs1,temp1, axis=0)
        elif y[i]==comp[1]:
            temp2=X[i,:]-np.transpose(mu2)
            Xsubs2 = np.append(Xsubs2,temp2, axis=0)
        elif y[i]==comp[2]:
            temp3=X[i,:]-np.transpose(mu3)
            Xsubs3 = np.append(Xsubs3,temp3, axis=0)
        elif y[i]==comp[3]:
            temp4=X[i,:]-np.transpose(mu4)
            Xsubs4 = np.append(Xsubs4,temp4, axis=0)
        else:
            temp5=X[i,:]-np.transpose(mu5)
            Xsubs5 = np.append(Xsubs5,temp5, axis=0)

    sX1 = np.hsplit(Xsubs1,2)
    X1 = sX1[0]
    X2 = sX1[1]

    sqX1 = np.square(X1)

    summed1 = sqX1.sum(axis=0)

    cov11=summed1/counts[0]

    sqX2 = np.square(X2)

    summed2 = sqX2.sum(axis=0)

    cov12=summed2/counts[0]

    X1X2=np.multiply(X1,X2)

    summed3=X1X2.sum(axis=0)

    cov13=summed3/counts[0]

    fcov1=np.array([cov11,cov13],dtype=float)
    scov1=np.array([cov13,cov12],dtype=float)
    covmat1=np.hstack((fcov1,scov1))

    sX2 = np.hsplit(Xsubs2,2)
    Y1 = sX2[0]
    Y2 = sX2[1]

    sqY1 = np.square(Y1)

    summed21 = sqY1.sum(axis=0)

    cov21=summed21/counts[1]

    sqY2 = np.square(Y2)

    summed22 = sqY2.sum(axis=0)

    cov22=summed22/counts[1]

    Y1Y2=np.multiply(Y1,Y2)

    summed23=Y1Y2.sum(axis=0)

    cov23=summed23/counts[1]

    fcov2=np.array([cov21,cov23],dtype=float)
    scov2=np.array([cov23,cov22],dtype=float)
    covmat2=np.hstack((fcov2,scov2))

    #Xsubs3

    sX3 = np.hsplit(Xsubs3,2)
    A1 = sX3[0]
    A2 = sX3[1]

    sqA1 = np.square(A1)

    summed31 = sqA1.sum(axis=0)

    cov31=summed31/counts[2]

    sqA2 = np.square(A2)

    summed32 = sqA2.sum(axis=0)

    cov32=summed32/counts[2]

    A1A2=np.multiply(A1,A2)

    summed33=A1A2.sum(axis=0)

    cov33=summed33/counts[2]

    fcov3=np.array([cov31,cov33],dtype=float)
    scov3=np.array([cov33,cov32],dtype=float)
    covmat3=np.hstack((fcov3,scov3))

    #XSubs4

    sX4 = np.hsplit(Xsubs4,2)
    B1 = sX4[0]
    B2 = sX4[1]

    sqB1 = np.square(B1)

    summed41 = sqB1.sum(axis=0)

    cov41=summed41/counts[3]

    sqB2 = np.square(B2)

    summed42 = sqB2.sum(axis=0)

    cov42=summed42/counts[3]

    B1B2=np.multiply(B1,B2)

    summed43=B1B2.sum(axis=0)

    cov43=summed43/counts[3]

    fcov4=np.array([cov41,cov43],dtype=float)
    scov4=np.array([cov43,cov42],dtype=float)
    covmat4=np.hstack((fcov4,scov4))

    #Xsubs5

    sX5 = np.hsplit(Xsubs5,2)
    C1 = sX5[0]
    C2 = sX5[1]

    sqC1 = np.square(C1)

    summed51 = sqC1.sum(axis=0)

    cov51=summed51/counts[4]

    sqC2 = np.square(C2)

    summed52 = sqC2.sum(axis=0)

    cov52=summed52/counts[4]

    C1C2=np.multiply(C1,C2)

    summed53=C1C2.sum(axis=0)

    cov53=summed53/counts[4]

    fcov5=np.array([cov51,cov53],dtype=float)
    scov5=np.array([cov53,cov52],dtype=float)
    covmat5=np.hstack((fcov5,scov5))

    covmats=np.vstack((covmat1,covmat2,covmat3,covmat4,covmat5))

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    pival=2*np.pi

    twopi=np.sqrt(pival)

    dec1 = covmat[0][0] * twopi

    a=np.hsplit(means,5)
    mu1=a[0]
    mu2=a[1]
    mu3=a[2]
    mu4=a[3]
    mu5=a[4]

    a=np.hsplit(Xtest,2)
    col1=a[0]
    col2=a[1]

    sub = col1-mu1[0]

    msub=np.square(sub)

    indec=(-0.5)*np.power(covmat[0][0],2)
    mult1=msub/indec

    num1=np.exp(mult1)

    px1=num1/dec1

    dec11 = covmat[1][1] * twopi

    sub11 = col2-mu1[1]

    msub11=np.square(sub11)

    indec11=(-0.5)*np.power(covmat[1][1],2)
    mult11=msub11/indec11

    num11=np.exp(mult11)

    px11=num11/dec11
    pclass1=px1*px11

    #class2

    sub21 = col1-mu2[0]

    msub21=np.square(sub21)

    mult21=msub21/indec

    num21=np.exp(mult21)

    px21=num21/dec1

    sub22 = col2-mu2[1]

    msub22=np.square(sub22)

    mult22=msub22/indec11

    num22=np.exp(mult22)

    px22=num22/dec11

    pclass2=px21*px22

     #class3

    sub31 = col1-mu3[0]
    msub31=np.square(sub31)

    mult31=msub31/indec

    num31=np.exp(mult31)

    px31=num31/dec1

    sub32 = col2-mu3[1]

    msub32=np.square(sub32)

    mult32=msub32/indec11

    num32=np.exp(mult32)

    px32=num32/dec11

    pclass3=px31*px32

    #class4

    sub41 = col1-mu4[0]

    msub41=np.square(sub41)

    mult41=msub41/indec

    num41=np.exp(mult41)

    px41=num41/dec1

    sub42 = col2-mu4[1]

    msub42=np.square(sub42)

    mult42=msub42/indec11

    num42=np.exp(mult42)

    px42=num42/dec11

    pclass4=px41*px42

    #class5

    sub51 = col1-mu5[0]

    msub51=np.square(sub51)

    mult51=msub51/indec

    num51=np.exp(mult51)

    px51=num51/dec1

    sub52 = col2-mu5[1]

    msub52=np.square(sub52)

    mult52=msub52/indec11

    num52=np.exp(mult52)

    px52=num52/dec11

    pclass5=px51*px52

    pclass=np.column_stack((pclass1,pclass2,pclass3,pclass4,pclass5))
    yp=[]

    max=np.argmax(pclass, axis=1)

    for i in range(0,max.shape[0]):
        if(max[i]==0):
             yp= np.append(yp,[1])
        elif(max[i]==1):
             yp= np.append(yp,[2])
        elif(max[i]==2):
             yp= np.append(yp,[3])
        elif(max[i]==3):
             yp= np.append(yp,[4])
        else:
             yp= np.append(yp,[5])

    ypred=np.reshape(yp,(ytest.shape[0],1))

    match=np.sum(ypred==ytest)

    total=ytest.shape[0]

    ac=np.true_divide(match,total)

    ac1=np.multiply(ac,100,dtype=float)
    acc=str(ac1)+'%'

    return acc,ypred



def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    pival=2*np.pi

    twopi=np.sqrt(pival)

    dec1 = covmats[0][0] * twopi

    a=np.hsplit(means,5)
    mu1=a[0]
    mu2=a[1]
    mu3=a[2]
    mu4=a[3]
    mu5=a[4]

    a=np.hsplit(Xtest,2)
    col1=a[0]
    col2=a[1]

    sub = col1-mu1[0]

    msub=np.square(sub)

    indec=(-0.5)*np.power(covmats[0][0],2)
    mult1=msub/indec

    num1=np.exp(mult1)

    px1=num1/dec1

    dec11 = covmats[1][1] * twopi

    sub11 = col2-mu1[1]

    msub11=np.square(sub11)
    indec11=(-0.5)*np.power(covmats[1][1],2)
    mult11=msub11/indec11

    num11=np.exp(mult11)

    px11=num11/dec11
    pclass1=px1*px11


     #class2

    sub21 = col1-mu2[0]

    dec21 = covmats[2][0] * twopi

    msub21=np.square(sub21)

    indec21=(-0.5)*np.power(covmats[2][0],2)

    mult21=msub21/indec21

    num21=np.exp(mult21)

    px21=num21/dec21

    sub22 = col2-mu2[1]

    dec22 = covmats[3][1] * twopi


    msub22=np.square(sub22)

    indec22=(-0.5)*np.power(covmats[3][1],2)

    mult22=msub22/indec22

    num22=np.exp(mult22)

    px22=num22/dec22

    pclass2=px21*px22

    #class2

    sub21 = col1-mu2[0]

    dec21 = covmats[2][0] * twopi

    msub21=np.square(sub21)

    indec21=(-0.5)*np.power(covmats[2][0],2)

    mult21=msub21/indec21

    num21=np.exp(mult21)

    px21=num21/dec21

    sub22 = col2-mu2[1]

    dec22 = covmats[3][1] * twopi


    msub22=np.square(sub22)

    indec22=(-0.5)*np.power(covmats[3][1],2)

    mult22=msub22/indec22

    num22=np.exp(mult22)

    px22=num22/dec22

    pclass2=px21*px22


    #class3

    sub31 = col1-mu3[0]

    dec31 = covmats[4][0] * twopi

    msub31=np.square(sub31)

    indec31=(-0.5)*np.power(covmats[4][0],2)

    mult31=msub31/indec31

    num31=np.exp(mult31)

    px31=num31/dec31

    sub32 = col2-mu3[1]

    dec32 = covmats[5][1] * twopi

    msub32=np.square(sub32)

    indec32=(-0.5)*np.power(covmats[5][1],2)

    mult32=msub32/indec32

    num32=np.exp(mult32)

    px32=num32/dec32

    pclass3=px31*px32

    #class4

    sub41 = col1-mu4[0]

    dec41 = covmats[6][0] * twopi

    msub41=np.square(sub41)

    indec41=(-0.5)*np.power(covmats[6][0],2)

    mult41=msub41/indec41

    num41=np.exp(mult41)

    px41=num41/dec41

    sub42 = col2-mu4[1]

    dec42 = covmats[7][1] * twopi

    msub42=np.square(sub42)

    indec42=(-0.5)*np.power(covmats[7][1],2)

    mult42=msub42/indec42

    num42=np.exp(mult42)

    px42=num42/dec42

    pclass4=px41*px42

     #class5

    sub51 = col1-mu5[0]

    dec51 = covmats[8][0] * twopi

    msub51=np.square(sub51)

    indec51=(-0.5)*np.power(covmats[8][0],2)

    mult51=msub51/indec51

    num51=np.exp(mult51)

    px51=num51/dec51

    sub52 = col2-mu5[1]

    dec52 = covmats[9][1] * twopi
    msub52=np.square(sub52)

    indec52=(-0.5)*np.power(covmats[9][1],2)

    mult52=msub52/indec52

    num52=np.exp(mult52)

    px52=num52/dec52

    pclass5=px51*px52

    pclass=np.column_stack((pclass1,pclass2,pclass3,pclass4,pclass5))
    yp=[]

    max=np.argmax(pclass, axis=1)

    for i in range(0,max.shape[0]):
        if(max[i]==0):
             yp= np.append(yp,[1])
        elif(max[i]==1):
             yp= np.append(yp,[2])
        elif(max[i]==2):
             yp= np.append(yp,[3])
        elif(max[i]==3):
             yp= np.append(yp,[4])
        else:
             yp= np.append(yp,[5])

    ypred = np.reshape(yp,(ytest.shape[0],1))

    match=np.sum(ypred==ytest)

    total=ytest.shape[0]

    ac=np.true_divide(match,total)

    ac1=np.multiply(ac,100,dtype=float)
    acc=str(ac1)+'%'

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD

    tx=np.transpose(X)
    inv1=np.dot(tx,X)

    inver=np.linalg.inv(inv1)

    term2=np.dot(tx,y);
    wprime=np.dot(inver,term2)

    xw=np.dot(X,wprime)

    diff=np.subtract(y,xw)

    tdiff=np.transpose(diff)

    term=np.dot(tdiff,diff)

    jw=term*(0.5)

    twprime=np.transpose(wprime)

    dw1=np.dot(twprime,tx)

    transdw1=np.transpose(dw1)

    sub1=transdw1-y

    final=np.multiply(sub1,X)

    gradjw=np.sum(final,axis=0)

    w=np.subtract(wprime,gradjw.reshape((X.shape[1],1)))

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    tx=np.transpose(X)

    inv1=np.dot(tx,X)

    id=np.identity(inv1.shape[0])

    lid=id*lambd

    sum1=inv1+lid

    finv=np.linalg.inv(sum1)

    term2=np.dot(finv,tx)

    w=np.dot(term2,y)

    return w



def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    tw=np.transpose(w)

    tx=np.transpose(Xtest)

    diff=np.dot(tw,tx)

    term=np.transpose(diff)

    sub1=ytest-term

    sub2=np.square(sub1)

    sum1=np.sum(sub2,axis=0)

    inner=sum1/Xtest.shape[0]

    rmse=np.sqrt(inner)

    # IMPLEMENT THIS METHOD
    return rmse





def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD

   xw=np.dot(X,w)

   diff=np.empty(y.shape)
   for i in range( y.shape[0] ):
       for j in range(y.shape[1]):
            diff[i][j] = y[i][j] - xw[i]

   diff2=np.dot(diff.T,diff)
   jw1=(0.5)*diff2

   lamw=lambd*(np.dot(w.T,w))

   jw2=(0.5)*lamw

   e=jw1+jw2

   x2=np.dot(X.T,X)

   x2w=np.dot(x2,w)

   emp=np.empty((y.shape[0],))

   for i in range(y.shape[0]):
       for j in range(y.shape[1]):
            emp[i]=y[i][j]

   grad2=np.dot(X.T,emp)

   lw=lambd*w

   error_grad=x2w-grad2+lw

   error=e.flatten()

   return error,error_grad



def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (p+1))
    # IMPLEMENT THIS METHOD

    Xd = np.ones((x.shape[0], p+1))
    for j in range (0,x.shape[0]):
        for i in range(0, p+1):
            Xd[j,i] = x[j] ** i
    return Xd



# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA

means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA

means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()


zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)



# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))




# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

plt.plot(lambdas,rmses3)


# Problem 4

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='BFGS', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

plt.plot(lambdas,rmses4)



# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))





