import numpy as np
#import tensorflow as tf
from t_prod import t_prod_me
import math
#import numpy as np
from numpy import linalg as la
from scipy.fftpack import fft,ifft
import datetime

def linsearch(X,D,S,Xome,T):
    a=np.power(np.linalg.norm(Xome*(X-S),ord=2),2)
    b=2*np.sum((Xome*(X-D))*(Xome*(S-X)))
   # print 'b',b
   # print 'a',a
    gamma=1.0/T
    if a!=0:
        if -b/(2*a)<0 or -b/(2*a)==0:
            gamma=1.0/T
        elif -b/(2*a)>1:
            gamma=1.0/T
        else:
            gamma=-b/(2*a)
    #print 'gamma',gamma
    return gamma

def t_svd(M):
	[n1 ,n2 ,n3] = M.shape
	D = np.zeros((n1 ,n2 ,n3), dtype = complex)
	D = fft(M)
	Uf = np.zeros((n1,n1,n3), dtype = complex)
	Thetaf = np.zeros((n1,n2,n3), dtype = complex)
	Vf = np.zeros((n2,n2,n3), dtype = complex)

	for i in range(n3):
		temp_U ,temp_Theta, temp_V = la.svd(D[: ,: ,i], full_matrices=True);
		Uf[: ,: ,i] = temp_U;
		Thetaf[:n2, :n2, i] = np.diag(temp_Theta)
		Vf[:, :, i] = temp_V;
	U = np.zeros((n1,n1,n3))
	Theta = np.zeros((n1,n2,n3))
	V = np.zeros((n2,n2,n3))
	U = ifft(Uf).real
	Theta = ifft(Thetaf).real
	V = ifft(Vf).real
	return U, Theta, V

#def Omegamatrix(Xx,samplenum):
#    (m,n,k)=np.shape(Xx)
 #   X=np.zeros((m,n,k))
 ##   Result = np.zeros((m, n, k))
    # = np.zeros(n)
  #  count=0
 #   for i in range(k):
 #       while(count<samplenum):
  #          locx = np.random.randint(m)
   #         locy=np.random.randint(n)
  #          X[locx][locy][i]=1
    #        Result[locx][locy][i]=Xx[locx][locy][i]
    #        count=count+1
   # return X,Result

def Omegamatrix(Xx,samplenum):
    (m,n)=np.shape(Xx)
    X=np.zeros((m,n))
    # = np.zeros(n)
    for i in range(m):
        loc=np.random.randint(n,size=samplenum)
       # print loc
        for j in loc:
            X[i][j]=1
    Result=X*Xx
    return X,Result

def dataprocess(X):
    (m, n,k) = np.shape(X)
    for j in range(k):
        for i in range(m):
            mean=sum(X[i,:,j])/len(X[i,:,j])
            X[i,:,j]=X[i,:,j]-mean
    return X

def rmse(target,prediction):
    error = []
    (m,n,k)=np.shape(target)
    for i in range(m):
        for j in range(n):
            for jj in range(k):
                error.append(target[i][j][jj] - prediction[i][j][jj])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    #print len(squaredError)
    rm=np.sqrt(sum(squaredError) / (m*n*k))
   # print sum(squaredError)
    return rm

def maxnorm(YY):
    (m,n,k)=np.shape(YY)

    maxx=0
    for i in range(k):
        temp=np.linalg.norm(YY[:,:,i],ord=2)
        if maxx<temp:
            maxx=temp
    return maxx


def update(Ome,n,k,v,lambda1,t,L,Yi,Di,gamma,dx):
    YA = np.zeros((m,n))
    Ai = np.zeros((m,n))
    AN=np.zeros((n,n))
    if t==0:
        Yiii=np.zeros((m,n))
    else:
        Yiii=Yi
    Ai=(Yiii-Di)*Ome
    #print 'Ai',Di
   # u=np.dot(np.dot(Ai,v),lambda1)
    u=(np.dot(Ai,v))*(1.0/lambda1)
    uu=[]
    uu.append(u)
    vv=[]
    vv.append(v)
    #print 'u',u
    #print uu.shape,'u'
    #print vv.shape,'v'
    #(1.0 / lambda1)
    #    print 'yiii',Yiiik*np.sqrt(dx)*
    Si=(k)*np.sqrt(dx)*np.dot(u,np.transpose(v))
    #print 'si',Si.shape
    gamma = linsearch(Yi, Di,Si,Ome,T )
    #print np.dot(u,np.transpose(v))
    Yite=(1.0-gamma)*Yiii-(k*gamma*np.sqrt(dx))*np.dot(u,np.transpose(v))

   # print 'yite;', np.dot(np.transpose(uu),vv)
    if np.linalg.norm(Yite*Ome,ord=2)!=0:
        if L/np.linalg.norm(Yite*Ome,ord=2)<1 :
            YA=(L/np.linalg.norm(Yite*Ome,ord=2))*Yite
        else:
            YA=Yite
    #YA=Yite
    Ai=(YA-Di)*Ome
   # print 'Ai:',Ai
    #Aii = []
    #Aii.append(Ai)i
    if t!=0:
       # Aii = []
       # Aii.append(Ai)
        #print Aii
        AN=np.dot(np.transpose(Ai),Ai)
    else:
        AN = np.dot(np.transpose(Ai), Ai)
   # print 'An:',AN.shape
    return YA,AN
#-------smart city data----
#path='ratings.txt'
#file=open(path,'r')
#useid=[]
#itemid=[]
#ratings=[]
#contextid=[]
#lines=file.readlines()
#for i in lines:
#    itemp=i.strip('\n').strip(' ').split(';')
 #   contextid.append(int(itemp[0]))
 #   itemid.append(int(itemp[1]))
  #  ratings.append(int(itemp[2]))
  #  useid.append(int(itemp[3]))
#itemmax=np.max(itemid)
#usemax=np.max(useid)
#contextmax=np.max(contextid)
#tensormat=np.zeros((usemax,itemmax,contextmax))
#print itemmax,usemax,contextmax
#for i in range(len(useid)):
 #       tensormat[useid[i]-1][itemid[i]-1][contextid[i]-1]=ratings[i]

 #----------moive data------
#import io
#import numpy as np
#import time
#import csv

#path = 'C:/Users/ChenKx/Desktop/movie/ratings.csv'
#path1 = 'C:/Users/ChenKx/Desktop\movie/tags.csv'
#file = open(path)
#file1 = open(path1)
#useid = []
#movieid = []
#ratings = []
#tagid = []
#csv_reader = csv.reader(file)
#csv_reader1 = csv.reader(file1)
#inu = 0
#for row in csv_reader:
 #   # itemp=row.strip('\n').strip(' ').split(';')
#    if inu != 0:
 #       itemp = row
  #      # print row
  #      if int(itemp[1]) < 1000:
  #          useid.append(int(itemp[0]))
   #         movieid.append(int(itemp[1]))
   #         ratings.append(float(itemp[2]))
   #         # print time.localtime(int(itemp[3]))[0]
   #         tagid.append(int(time.localtime(int(itemp[3]))[0]))
  #  inu = inu + 1
  #  # useid.append(int(itemp[3]))
#moviemax = np.max(movieid)
#usemax = np.max(useid)
#tagmax = np.max(tagid) - np.min(tagid)
## contextmax=np.max(contextid)
#print moviemax, usemax, tagmax
#tensormat = np.zeros((usemax, moviemax, tagmax))

#for i in range(len(useid)):
 #   tensormat[useid[i] - 1][movieid[i] - 1][tagid[i] - np.min(tagid) - 1] = ratings[i]
##print len(np.nonzero(tensormat)[0])
#data=tensormat


#-------music-----
#import numpy as np
#import csv
#import io
#path='C:/Users/ChenKx/Desktop/music/artists.csv'
#file = open(path)
#csv_reader = csv.reader(file)
#artist=[]
#country=[]
#tag=[]
#ratings1=[]
#ratings2=[]
#i=0
#for row in csv_reader:
#    if i!=0 and i<=1000:
#        if row[1].isalpha():
 #           artist.append(row[1])
#        country.append(row[3])
#        for j in row[5].split(';'):
 ##           if all(jj.isalpha() or jj==' ' for jj in j.strip(' ').strip('\n')):
  #              tag.append(j.strip(' ').strip('\n'))
 #             #  print len(tag)
 #       if row[7]!='':
 #           ratings1.append(int(row[7]))
 #       if row[8]!='':
 #           ratings2.append(int(row[8]))
 #   i=i+1

#artist=set(artist)
#country=set(country)
#tag=set(tag)
##print tag
#artistmax=len(artist)
#countrymax=len(country)
#tagmax=len(tag)
#ratings1max=np.max(ratings1)
#ratings1min=np.min(ratings1)
#ratings2max=np.max(ratings2)
#ratings2min=np.min(ratings2)
#qujian1=(ratings1max-ratings1min)
#qujian2=(ratings2max-ratings2min)
#print qujian1,qujian2
#i=0
#aa=np.zeros((artistmax,countrymax,tagmax))
#print artistmax,countrymax,tagmax
#file = open(path)
#csv_reader = csv.reader(file)
#aii=[]
#cii=[]
#tii=[]
#aaa=[]
#for row in csv_reader:
    #print row[1]
    #print list(artist)
 #   if i!=0 and i<1000:
      # try:
           # if row[1].isalpha():
           #     ai= list(artist).index(row[1])
         #   ci=list(country).index(row[3])
         #   aii.append(ai)
         #   cii.append(ci)
        #  #  print ai,ci
       # except ValueError:
        #    print ''
      #  if row[7] != '':
         #   rating1=(int(row[7])-ratings1min)/1000000.0
       # if row[8] != '':
       #     rating2=(int(row[8])-ratings2min)/1000000000.0
     # #  print 'r12',rating1,rating2
       # for j in row[5].split(';'):
          # # print j
           # if j.strip(' ').strip('\n').isalpha():
              # # print j
                 #   ti=list(tag).index(j.strip(' ').strip('\n'))
               #     tii.append(ti)
             #       #print ti
            #       # print rating1 + rating2
       #             aaa.append(rating1 + rating2)
  #  #print len(np.nonzero(aa))
  #  i=i+1
#print len(aii),len(cii),len(tii),len(aaa)
#for i,j,k in zip(range(len(aii)),range(len(cii)),range(len(tii))):
  #  #print aii[i],cii[j],tii[k]
  #  aa[aii[i],cii[j],tii[k]]=int(aaa[k])

#data=aa


            # print country

    #print tag

#--------syntic data-------
m=200
n=200
k=20
r=1

data=t_prod_me(np.random.rand(m,r,k),np.random.rand(r,n,k))

#data=dataprocess(data)
#print np.real(data)
#(du,ds,dv)=t_svd(data)
#dsmax=np.max(ds)
#print 'dsmax',dsmax
(n_sample,n_feature,n_serve)=data.shape
rate=0.3
samplenumber=rate*n_feature
datainput=np.zeros((n_sample,n_feature,n_serve))
Xomega=np.zeros((n_sample,n_feature,n_serve))
for nsi in range(n_serve):
    Xomega[:,:,nsi],datainput[:,:,nsi]=Omegamatrix(data[:,:,nsi],int(samplenumber))

(m,n,k)=np.shape(datainput)
#kk can be changed
kk=200 
L =float(maxnorm(datainput))
#print 'data',datainput[:,:,1]
print 'L',L
#-----------------------------------------

#------------------------------glocal-----
di=[]
delta=1.0/np.power(10,6)
#eplison=[1]
eplison=[0.1,1.0,2.0,5.0,12.0]
#_sample=5000
#n_feature=40
#Ti=[i for i in range(5,20,10)]
T = 20
belta = 10
tt=[]
for ep in eplison:
    #start=datetime.datetime.now()
    sigma=2*L*np.sqrt(2*T*np.log10(1/delta))/(ep*1000)
    print 'sigma:',sigma
    #print '1:',np.power(L,2)
    vv=np.zeros((n,n,k))
    #vv=np.zeros((n,n))
    lamdasqrt=np.zeros((n,n,k))
    #lamsqrt=np.zeros((n*k,n))

    Y=np.zeros((m,n,k))
    for t in range(T):
        #print t
        #print Y
        W=np.zeros((n,n,k))
        #for ii in range(len(lamda)):

        gamma=1.0/T

        dx=np.sqrt(k)
        #lamdp=math.sqrt(8 * np.power(sigma, 1) * (
        #np.sum(m + n + k) * np.log10(float(2 * (m + n + k)) / np.log10(3.0 / 2)) + np.log10(2.0 / belta)))
        #print np.sqrt(lamdp)
       # print '3:',math.sqrt(sigma*np.log10(float(n)/belta))*math.pow(n,1.0/4)
        for i in range(k):
            Di=datainput[:,:,i]
            #print 'Di',Di
            Xomegai=Xomega[:,:,i]
            #lamda=np.zeros((n,n))
            lamma=1
            #lamdaindex=lamdasqrt.index(max(lamdasqrt))
            #lamdaindex=np.where(lamdasqrt[:,:,i]==np.max(lamdasqrt[:,:,i]))
            #print lamdaindex[1][0]
           # lamda=np.max(lamdasqrt[:,:,i])
           # print lamda
           # if lamda>0:
           #     lamda=np.sqrt(lamda)
           # else:
            #    lamda=np.sqrt(np.abs(lamda))
            #lamda=np.sqrt(lamda/k+np.sqrt(lamdp))
          #  print lamda
            #print lamdasqrt[:,:,i]
            #for lamtemp in range(n):
            #    for lamtempj in range(n):
            #        if lamdasqrt[lamtemp,lamtempj,i]>0:
            #            lamda[lamtemp][lamtempj]= (np.sqrt(lamdasqrt[lamtemp, lamtempj, i])+lamdp)
            #        else:
            #            lamda[lamtemp][lamtempj]=0

            #print 8*np.power(sigma,2)*(np.sum(m+n+k)*np.log10(float(2*(m+n+k))/np.log10(3.0/2))+np.log10(2.0/belta))
            #print 'lamda',lamda
#            lamdex=np.where(lamsqrt==np.max(lamsqrt))
            #print np.max(lamsqrt)
            #print vv.shape
            #v=vv
            #print vv
            v = np.real(vv[ :,:,i])#[:,lamdaindex[1][0]]
            #v=np.real(vv[0,i*n:(i+1)*n])
            #print v
            #print 'v',v
            Yii,AN=update(Xomegai, n, kk, v, lamma, t, L, Y[:,:,i], Di, gamma, dx)
            #Yii,AN=update(Xomegai,n,k,v,lambda1,T,t,L,Y[:,:,i],Di)
            #print 'Yii',Yii:,
            #print 'AN',AN
            Y[:,:,i]=Yii
            W[:,:,i]=AN
        for it in range(n):
            for jt in range(n):
                for kt in range(k):
                    W[it][jt][kt]=W[it][jt][kt]+ np.random.normal(0,np.power(sigma,1))
                #print np.random.normal(0,np.power(sigma,1))
       # print 'W:',W
       # Wfold=np.zeros((n*k,n))
        #for j in range(k):
        #    for jjj in range(n):
         #       Wfold[j*n+jjj,:]=W[jjj,:,j]
        ##print W
        #uu,lamsqrt,vv=np.linalg.svd(Wfold)
        uu,lamdasqrt,vv = t_svd(W)
        lamsum=np.zeros((n,n))
        for li in range(k):
            #print lamdasqrt[:,:,li]
            lamsum =lamsum+lamdasqrt[:,:,li]
        lamma=np.sqrt(np.max(lamsum)/k)-np.sqrt((n*sigma)/k)*np.power((m+n+k),1/4)
       # print '1',np.max(lamsum)
        #print '2',n*sigma
        #print lamma
        #end=datetime.datetime.now()
        #tt.append(end-start)
       # ui,lami,vi=np.linalg.svd(W[:,:,1])
        #print 'vi',vi
        #print 'v0i',vv[:,:,1]
       # print 'wvv', np.dot(W[:,:,1],vv[:,:,1])
       # print 'wvi', np.dot(W[:,:,1],vi)
        #vvv=np.zeros((n,n))
       # for iiii in range(n):
        #    for jjjj in range(n):
         #       vvv[iiii,jjjj]=vv[jjjj,iiii,1]
    #    print 'tsvd',t_prod(uu,lamdasqrt)*vv[:,:,1]
      #  print 'w',W[1,1,1]
       # print np.transpose(W).shape
        #$WT=np.zeros((n,n,k))
       # WT[:,:,0]=np.transpose(W[:,:,0])
      #  for wti in range(k-1):
       #     WT[:,:,wti+1]=np.transpose(W[:,:,k-(wti+1)])
       # u1,lamdasqrt1,v1=t_svd(t_prod(WT,W))
        #print 'vv',vv.all()
        #print 'v1',v1.all()
        #if vv.all()==v1.all():
         #   print 'true'
        #else:
        #    print 'false'
       # print 'n',np.dot(np.dot(uu[:,:,1],lamdasqrt[:,:,1]),np.transpose(vv[:,:,1]))
        #print 'w',W[:,:,1]
        #print 'vv',vv
       # print 'lamdasqrt',lamdasqrt
       # print vv.shape


        #print 'lamdamx:',lamdasqrt[lamdamax]

   # print 'Y:',(Y)[:,:,1]
   # print 'Y:', (Y*Xomega)[:, :, 1]
    #Y=np.zeros((m,n,k))
   # print Y-data
    di.append(rmse(Y*Xomega,datainput))
    print di
#print tt
#plt.plot(eplison, di, '-r^');
#plt.xlabel('Epsilon');
#plt.ylabel('RMSE');
#plt.grid(color='black',linewidth='0.3',linestyle='--')
#plt.show()
