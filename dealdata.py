import os
import numpy as np

def reddit():
    path = 'C:/Users/yangjia/Desktop/federated/soc-redditHyperlinks-title.tsv'
    file = open(path, 'r')
    useid=[]
    itemid=[]
    ratings=[]
    contextid=[]
    lines=file.readlines()
    for i in lines:
       itemp=i.strip('\n').strip(' ').split(';')
       contextid.append(int(itemp[0]))
       itemid.append(int(itemp[1]))
       ratings.append(int(itemp[2]))
       useid.append(int(itemp[3]))
    itemmax=np.max(itemid)
    usemax=np.max(useid)
    contextmax=np.max(contextid)
    tensormat=np.zeros((usemax,itemmax,contextmax))
    print itemmax,usemax,contextmax
    for i in range(len(useid)):
           tensormat[useid[i]-1][itemid[i]-1][contextid[i]-1]=ratings[i]

    return tensormat

def dealmovie():
    path = 'C:/Users/yangjia/Desktop/movie/ratings.csv'
    file = open(path)
    #  file1 = open(path1)
    useid = []
    movieid = []
    ratings = []
    timeid = []
    csv_reader = csv.reader(file)
    #    csv_reader1 = csv.reader(file1)
    inu = 0
    for row in csv_reader:
        # itemp=row.strip('\n').strip(' ').split(';')
        if inu != 0:
            itemp = row
            # if int(itemp[1]) < 50:
            useid.append(int(itemp[0]))
            movieid.append(int(itemp[1]))
            ratings.append(float(itemp[2]))
            # print time.localtime(int(itemp[3]))[0]
            timeid.append(int(time.localtime(int(itemp[3]))[0]))
        inu = inu + 1
        # useid.append(int(itemp[3]))
    usernameset = set(useid)
    timeset = set(timeid)
    movieset = set(movieid)
    m = len(usernameset)
    print 'm', m
    nodecount = {}
    moviecount = {}
    timecount = {}
    for uitem in usernameset:
        nodecount[uitem] = int(useid.count(uitem))
    for titme in timeset:
        timecount[titme] = int(timeid.count(titme))
    for mitme in movieset:
        moviecount[mitme] = int(movieid.count(mitme))
        # nodecount.append(nodename.count(item))
    # nodecount.sort(key=takeSecond)
    # print nodecount.keys()
    # nodecountkey=nodecount.keys()
    nodecount = sorted(nodecount.items(), key=lambda x: x[1], reverse=True)
    timecount = sorted(timecount.items(), key=lambda x: x[1], reverse=True)
    moviecount = sorted(moviecount.items(), key=lambda x: x[1], reverse=True)
    # print nodecount[0][0]
    print len(moviecount), len(nodecount), len(timecount)
    nodenum = 200  # len(nodecount)
    movienum = 200
    timenum = len(timecount)
    nodename = []
    moviename = []
    ts = []
    # nodecountkey=nodecount.keys()
    for i in range(nodenum):
        nodename.append(nodecount[i][0])
    for i in range(timenum):
        ts.append(timecount[i][0])
    for i in range(movienum):
        moviename.append(moviecount[i][0])
    # ts=list(set(timesamp))
    k = timenum
    # print timecount
    n = movienum
    # for i in timesamp:
    #   if
    m = nodenum
    nodefeature = np.zeros((k, m, n))
    Xomeganew = np.zeros((k, m, n))
    file.close()
    file = open(path)
    csv_reader = csv.reader(file)
    ii = 0
    icount = 0
    # print nodename,moviename,ts
    for row in csv_reader:
        # itemp=row.strip('\n').strip(' ').split(';')
        ttime = time.localtime(int(itemp[3]))[0]
        # print int(itemp[0]) in nodename
        if ii != 0:
            itemp = row
            if int(itemp[0]) in nodename:
                if int(itemp[1]) in moviename:
                    if int(ttime) in ts:
                        icount = icount + 1
                        # print ts.index(ttime), nodename.index(int(itemp[0])), moviename.index(int(itemp[1]))
                        nodefeature[ts.index(int(ttime))][nodename.index(int(itemp[0]))][
                            moviename.index(int(itemp[1]))] = float(itemp[2])
                        Xomeganew[ts.index(int(ttime))][nodename.index(int(itemp[0]))][
                            moviename.index(int(itemp[1]))] = 1
        ii = ii + 1
    # tensormat[useid[i] - 1][movieid[i] - 1][tagid[i] - np.min(tagid) - 1] = ratings[i]
    # #print len(np.nonzero(tensormat)[0])
    data = nodefeature
    return data,Xomeganew

  import matplotlib
  from PIL import Image

  def ImagetoMatrix(filename):
      im=Image.open(filename)
      width,height=im.size
     # print width,height
      im=im.convert("L")
      data=im.getdata()
      data=np.matrix(data,dtype='float')
      new_data=np.reshape(data,(height,width))
      return new_data

  def MatrixToImage(data):
      data=data
      new_im=Image.fromarray(np.uint8(data))
      new_im=new_im.convert("L")
      return new_im

  import numpy as np
  import csv
  import io

  def dealmusic():
    path='C:/Users/yangjia/Desktop/music/artists.csv'
    file = open(path)
    csv_reader = csv.reader(file)
    artist=[]
    country=[]
    tag=[]
    ratings1=[]
    ratings2=[]
    i=0
    for row in csv_reader:
       if i!=0 and i<=1000:
           if row[1].isalpha():
               artist.append(row[1])
           country.append(row[3])
           for j in row[5].split(';'):
     #           if all(jj.isalpha() or jj==' ' for jj in j.strip(' ').strip('\n')):
                   tag.append(j.strip(' ').strip('\n'))
                 #  print len(tag)
           if row[7]!='':
               ratings1.append(int(row[7]))
           if row[8]!='':
               ratings2.append(int(row[8]))
       i=i+1

    artist=set(artist)
    country=set(country)
    tag=set(tag)
    #print tag
    artistmax=len(artist)
    countrymax=len(country)
    tagmax=len(tag)
    ratings1max=np.max(ratings1)
    ratings1min=np.min(ratings1)
    ratings2max=np.max(ratings2)
    ratings2min=np.min(ratings2)
    qujian1=(ratings1max-ratings1min)
    qujian2=(ratings2max-ratings2min)
    print qujian1,qujian2
    i=0
    aa=np.zeros((artistmax,countrymax,tagmax))
    print artistmax,countrymax,tagmax
    file = open(path)
    csv_reader = csv.reader(file)
    aii=[]
    cii=[]
    tii=[]
    aaa=[]
    for row in csv_reader:
        print row[1]
        print list(artist)
       if i!=0 and i<1000:
          try:
               if row[1].isalpha():
                   ai= list(artist).index(row[1])
               ci=list(country).index(row[3])
               aii.append(ai)
               cii.append(ci)
             #  print ai,ci
           except ValueError:
               print ''
           if row[7] != '':
               rating1=(int(row[7])-ratings1min)/1000000.0
           if row[8] != '':
               rating2=(int(row[8])-ratings2min)/1000000000.0
         #  print 'r12',rating1,rating2
           for j in row[5].split(';'):
              # print j
               if j.strip(' ').strip('\n').isalpha():
                  # print j
                       ti=list(tag).index(j.strip(' ').strip('\n'))
                       tii.append(ti)
                       #print ti
                      # print rating1 + rating2
                       aaa.append(rating1 + rating2)
       #print len(np.nonzero(aa))
       i=i+1
    print len(aii),len(cii),len(tii),len(aaa)
    for i,j,k in zip(range(len(aii)),range(len(cii)),range(len(tii))):
       #print aii[i],cii[j],tii[k]
       aa[aii[i],cii[j],tii[k]]=int(aaa[k])

    data=aa
    return data










