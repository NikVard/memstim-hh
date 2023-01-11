#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:57:48 2018

@author: aussel
"""
from brian2 import *
import ast
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

close('all')

raster_EC_exc_i=[]
raster_DG_exc_i=[]
raster_CA3_exc_i=[]
raster_CA1_exc_i=[]

raster_EC_exc_t=[]
raster_DG_exc_t=[]
raster_CA3_exc_t=[]
raster_CA1_exc_t=[]

raster_EC_inh_i=[]
raster_DG_inh_i=[]
raster_CA3_inh_i=[]
raster_CA1_inh_i=[]

raster_EC_inh_t=[]
raster_DG_inh_t=[]
raster_CA3_inh_t=[]
raster_CA1_inh_t=[]

positions_EC_exc=[]
positions_EC_inh=[]
positions_DG_exc=[]
positions_DG_inh=[]
positions_CA3_exc=[]
positions_CA3_inh=[]
positions_CA1_exc=[]
positions_CA1_inh=[]

#file_i=open('harry/raster_2/raster_exc_i.txt','r')
#file_t=open('harry/raster_2/raster_exc_t.txt','r')
#for data in file_i:
#    i_exc=data.split(',')[:-1]
#    i_exc=[int(k) for k in i_exc]
#del data
#
#for data in file_t:
#    t_exc=data.split(',')[:-1]
#    t_exc=[float(k) for k in t_exc]
#del data
#
#file_i.close()
#file_t.close()
#
#for ind in range(len(i_exc)):
#    if i_exc[ind]<10000:
#        raster_CA1_exc_i.append(i_exc[ind])
#        raster_CA1_exc_t.append(t_exc[ind]+6)
#    elif i_exc[ind]<11000:
#        raster_CA3_exc_i.append(i_exc[ind]-10000)
#        raster_CA3_exc_t.append(t_exc[ind]+6)
#    elif i_exc[ind]<21000 :
#        raster_DG_exc_i.append(i_exc[ind]-11000)
#        raster_DG_exc_t.append(t_exc[ind]+6)
#    else :
#        raster_EC_exc_i.append(i_exc[ind]-21000)
#        raster_EC_exc_t.append(t_exc[ind]+6)
#
#del t_exc,i_exc
#
##figure()
##plot(raster_CA1_exc_t,raster_CA1_exc_i,'ro')
#
#
#file_i=open('harry/raster_2/raster_inh_i.txt','r')
#file_t=open('harry/raster_2/raster_inh_t.txt','r')
#for data in file_i:
#    i_inh=data.split(',')[:-1]
#    i_inh=[int(k) for k in i_inh]
#del data
#
#for data in file_t:
#    t_inh=data.split(',')[:-1]
#    t_inh=[float(k) for k in t_inh]
#del data
#
#file_i.close()
#file_t.close()
#
#for ind in range(len(i_inh)):
#    if i_inh[ind]<1000:
#        raster_CA1_inh_i.append(i_inh[ind])
#        raster_CA1_inh_t.append(t_inh[ind]+6)
#    elif i_inh[ind]<1100:
#        raster_CA3_inh_i.append(i_inh[ind]-1000)
#        raster_CA3_inh_t.append(t_inh[ind]+6)
#    elif i_inh[ind]<1200 :
#        raster_DG_inh_i.append(i_inh[ind]-1100)
#        raster_DG_inh_t.append(t_inh[ind]+6)
#    else :
#        raster_EC_inh_i.append(i_inh[ind]-1200)
#        raster_EC_inh_t.append(t_inh[ind]+6)
#
#del t_inh,i_inh

def make_raster(filename_i,filename_t):
    tmax=0.8
    i=[]
    t=[]
    file_i=open(filename_i,'r')
    file_t=open(filename_t,'r')
    for data in file_i:
        i=data.split(',')[:-1]
        i=[int(k) for k in i]
    for data in file_t:
        t=data.split(',')[:-1]
        for j in range(len(t)):
            if t[j][-2]=='m': #ms
                t[j]=float(t[j][:-3])/1000
            else : #s
                t[j]=float(t[j][:-1])
    file_i.close()
    file_t.close()
    raster_i=[]
    raster_t=[]
    for ind in range(len(i)):
        raster_i.append(i[ind])
        raster_t.append(t[ind])
        if t[ind]>=tmax:
#            print('stop')
            return raster_i[:-1],raster_t[:-1]
    return raster_i,raster_t

#path='results_2019-10-01 09:14:17.368983'
path='results_2019-10-01 11:13:56.902280/results_[0]'
raster_CA1_exc_i,raster_CA1_exc_t=make_raster(path+'/rasters/raster_CA1_exc_i.txt',path+'/rasters/raster_CA1_exc_t.txt')
raster_CA3_exc_i,raster_CA3_exc_t=make_raster(path+'/rasters/raster_CA3_exc_i.txt',path+'/rasters/raster_CA3_exc_t.txt')
raster_DG_exc_i,raster_DG_exc_t=make_raster(path+'/rasters/raster_DG_exc_i.txt',path+'/rasters/raster_DG_exc_t.txt')
raster_EC_exc_i,raster_EC_exc_t=make_raster(path+'/rasters/raster_EC_exc_i.txt',path+'/rasters/raster_EC_exc_t.txt')

raster_CA1_inh_i,raster_CA1_inh_t=make_raster(path+'/rasters/raster_CA1_inh_i.txt',path+'/rasters/raster_CA1_inh_t.txt')
raster_CA3_inh_i,raster_CA3_inh_t=make_raster(path+'/rasters/raster_CA3_inh_i.txt',path+'/rasters/raster_CA3_inh_t.txt')
raster_DG_inh_i,raster_DG_inh_t=make_raster(path+'/rasters/raster_DG_inh_i.txt',path+'/rasters/raster_DG_inh_t.txt')
raster_EC_inh_i,raster_EC_inh_t=make_raster(path+'/rasters/raster_EC_inh_i.txt',path+'/rasters/raster_EC_inh_t.txt')

def make_pos(filename):
    file_pos=open(filename,'r')
    positions=[]
    ind=0
    for data in file_pos:
        pos=data[1:-3].split()
        pos=[float(k) for k in pos if k!='']
        positions.append(pos)
        ind+=1
    file_pos.close()
    return positions
positions_CA1_exc=make_pos(path+'/positions/positions_CA1_exc.txt')
positions_CA3_exc=make_pos(path+'/positions/positions_CA3_exc.txt')
positions_DG_exc=make_pos(path+'/positions/positions_DG_exc.txt')
positions_EC_exc=make_pos(path+'/positions/positions_EC_exc.txt')

positions_CA1_inh=make_pos(path+'/positions/positions_CA1_inh.txt')
positions_CA3_inh=make_pos(path+'/positions/positions_CA3_inh.txt')
positions_DG_inh=make_pos(path+'/positions/positions_DG_inh.txt')
positions_EC_inh=make_pos(path+'/positions/positions_EC_inh.txt')


#file_pos_e=open('harry/raster_2/positions_n_exc.txt','r')
#ind=0
#for data in file_pos_e:
#    pos=data[1:-3].split()
#    pos=[float(k) for k in pos]
#    if ind<10000:
#        positions_CA1_exc.append(pos)
#    elif ind<11000:
#        positions_CA3_exc.append(pos)
#    elif ind<21000:
#        positions_DG_exc.append(pos)
#    else :
#        if len(pos)==3:
#            positions_EC_exc.append(pos)
#    ind+=1
#del data
#
#file_pos_e.close()
#
#file_pos_i=open('harry/raster_2/positions_n_inh.txt','r')
#ind=0
#for data in file_pos_i:
#    pos=data[1:-3].split()
#    pos=[float(k) for k in pos]
#    if ind<1000:
#        positions_CA1_inh.append(pos)
#    elif ind<1100:
#        positions_CA3_inh.append(pos)
#    elif ind<1200:
#        positions_DG_inh.append(pos)
#    else :
#        if len(pos)==3:
#            positions_EC_inh.append(pos)
#    ind+=1
#del data
#
#file_pos_i.close()

#figure()
#plot(raster_CA1_inh_t,raster_CA1_inh_i,'ro')
minlim=0.5
maxlim=0.8

#figure()
#subplot(411)
#plot(raster_CA3_inh_t,raster_CA3_inh_i,'ro',markersize=2)
#xlim(minlim,maxlim)
#subplot(412)
#plot(raster_CA1_inh_t,raster_CA1_inh_i,'ro',markersize=2)
#xlim(minlim,maxlim)
#ylim(1,100)
#subplot(413)
#plot(raster_CA3_exc_t,raster_CA3_exc_i,'ro',markersize=2)
#xlim(minlim,maxlim)
#ylim(1,100)
#subplot(414)
#plot(raster_CA1_exc_t,raster_CA1_exc_i,'ro',markersize=2)
#xlim(minlim,maxlim)
#ylim(1,100)

raster_CA3_inh_t=array(raster_CA3_inh_t)
raster_CA3_inh_i=array(raster_CA3_inh_i)
raster_CA3_exc_t=array(raster_CA3_exc_t)
raster_CA3_exc_i=array(raster_CA3_exc_i)
raster_CA1_inh_t=array(raster_CA1_inh_t)
raster_CA1_inh_i=array(raster_CA1_inh_i)
raster_CA1_exc_t=array(raster_CA1_exc_t)
raster_CA1_exc_i=array(raster_CA1_exc_i)

#positions_EC_exc=array(positions_EC_exc[:2956]+positions_EC_exc[2957:])
positions_EC_exc=array(positions_EC_exc)
positions_EC_inh=array(positions_EC_inh)
positions_DG_exc=array(positions_DG_exc)
positions_DG_inh=array(positions_DG_inh)
positions_CA3_exc=array(positions_CA3_exc)
positions_CA3_inh=array(positions_CA3_inh)
positions_CA1_exc=array(positions_CA1_exc)
positions_CA1_inh=array(positions_CA1_inh)

figure()
subplot(411)
plot(raster_EC_exc_t,raster_EC_exc_i,'r.')
subplot(412)
plot(raster_DG_exc_t,raster_DG_exc_i,'r.')
subplot(413)
plot(raster_CA3_exc_t,raster_CA3_exc_i,'r.')
subplot(414)
plot(raster_CA1_exc_t,raster_CA1_exc_i,'r.')

figure()
subplot(411)
plot(raster_EC_inh_t,raster_EC_inh_i,'r.')
subplot(412)
plot(raster_DG_inh_t,raster_DG_inh_i,'r.')
subplot(413)
plot(raster_CA3_inh_t,raster_CA3_inh_i,'r.')
subplot(414)
plot(raster_CA1_inh_t,raster_CA1_inh_i,'r.')

#figure()
#plot(raster_CA3_inh_t,array(raster_CA3_inh_i)+300,'bo',markersize=2)
#plot(array(raster_CA3_exc_t[where(raster_CA3_exc_i<100)]),array(raster_CA3_exc_i[where(raster_CA3_exc_i<100)])+200,'ro',markersize=2)
#xlim(minlim,maxlim)
#plot(array(raster_CA1_inh_t[where(raster_CA1_inh_i<100)]),array(raster_CA1_inh_i[where(raster_CA1_inh_i<100)])+100,'bo',markersize=2)
#plot(array(raster_CA1_exc_t[where(raster_CA1_exc_i<100)]),array(raster_CA1_exc_i[where(raster_CA1_exc_i<100)]),'ro',markersize=2)
#xlim(minlim,maxlim)
#ylim(1,400)
#xlim(0.425,0.65)
#
#
##Mesure de la cohérence :
#def coherence(raster_i,raster_t,N,wind=5*msecond,numpairs=2000):
#    raster_binned=zeros((int(2*second/wind),N))
#
#    for ind in range(len(raster_t)):
#        raster_binned[int(raster_t[ind]*second/wind),raster_i[ind]]=1
#
#    kappa_sum=0
#    for k in range(numpairs):
#        i=randint(N)
#        j=randint(N)
#        while j==i:
#            j=randint(N)
#        Si=sum(raster_binned[:,i])
#        Sj=sum(raster_binned[:,j])
#        Sij=sum(dot(raster_binned[:,i],raster_binned[:,j]))
##        print(Si,Sj,Sij)
##        print(Sij/sqrt(Si*Sj))
#        if Si==0 or Sj==0 :
#            kappa_sum+=0
#        else :
#            kappa_sum+=Sij/sqrt(Si*Sj)
#    return kappa_sum/numpairs
#
#
##extraction d'une fenêtre du raster :
#def fen(raster_i,raster_t,tmin,tmax):
#    new_i=[]
#    new_t=[]
#    for ind in range(len(raster_t)):
#        if raster_t[ind]>=tmin and raster_t[ind]<=tmax:
#            new_i.append(raster_i[ind])
#            new_t.append(raster_t[ind]-tmin)
#    return new_i,new_t
#
#
##extrait_ca3_i,extrait_ca3_t=fen(raster_CA3_exc_i,raster_CA3_exc_t,0.6,0.8)
##print(coherence(extrait_ca3_i,extrait_ca3_t,1000))
#
#
#
#print('Nb spike moyen')
#print('EC exc: '+str(len(raster_EC_exc_i)/10000/2))
#print('EC inh: '+str(len(raster_EC_inh_i)/1000/2))
#print('DG exc: '+str(len(raster_DG_exc_i)/10000/2))
#print('DG inh: '+str(len(raster_DG_inh_i)/100/2))
#print('CA3 exc: '+str(len(raster_CA3_exc_i)/1000/2))
#print('CA3 inh: '+str(len(raster_CA3_inh_i)/100/2))
#print('CA1 exc: '+str(len(raster_CA1_exc_i)/10000/2))
#print('CA1 inh: '+str(len(raster_CA1_inh_i)/1000/2))
#
#Nspike_EC_exc=zeros(10000)
#Nspike_EC_inh=zeros(1000)
#Nspike_DG_exc=zeros(10000)
#Nspike_DG_inh=zeros(100)
#Nspike_CA3_exc=zeros(1000)
#Nspike_CA3_inh=zeros(100)
#Nspike_CA1_exc=zeros(10000)
#Nspike_CA1_inh=zeros(1000)
#
##ECe_i,ECe_t=fen(raster_EC_exc_i,raster_EC_exc_t,0,0.7)
##ECi_i,ECi_t=fen(raster_EC_inh_i,raster_EC_inh_t,0,0.7)
##DGe_i,DGe_t=fen(raster_DG_exc_i,raster_DG_exc_t,0,0.7)
##DGi_i,DGi_t=fen(raster_DG_inh_i,raster_DG_inh_t,0,0.7)
##CA3e_i,CA3e_t=fen(raster_CA3_exc_i,raster_CA3_exc_t,0,0.7)
##CA3i_i,CA3i_t=fen(raster_CA3_inh_i,raster_CA3_inh_t,0,0.7)
##CA1e_i,CA1e_t=fen(raster_CA1_exc_i,raster_CA1_exc_t,0,0.7)
##CA1i_i,CA1i_t=fen(raster_CA1_inh_i,raster_CA1_inh_t,0,0.7)
#ECe_i,ECe_t=fen(raster_EC_exc_i,raster_EC_exc_t,1.3,1.6)
#ECi_i,ECi_t=fen(raster_EC_inh_i,raster_EC_inh_t,1.3,1.6)
#DGe_i,DGe_t=fen(raster_DG_exc_i,raster_DG_exc_t,1.3,1.6)
#DGi_i,DGi_t=fen(raster_DG_inh_i,raster_DG_inh_t,1.3,1.6)
#CA3e_i,CA3e_t=fen(raster_CA3_exc_i,raster_CA3_exc_t,1.3,1.6)
#CA3i_i,CA3i_t=fen(raster_CA3_inh_i,raster_CA3_inh_t,1.3,1.6)
#CA1e_i,CA1e_t=fen(raster_CA1_exc_i,raster_CA1_exc_t,1.3,1.6)
#CA1i_i,CA1i_t=fen(raster_CA1_inh_i,raster_CA1_inh_t,1.3,1.6)
#for k in ECe_i:
#    Nspike_EC_exc[k]+=1
#for k in ECi_i:
#    Nspike_EC_inh[k]+=1
#for k in DGe_i:
#    Nspike_DG_exc[k]+=1
#for k in DGi_i:
#    Nspike_DG_inh[k]+=1
#for k in CA3e_i:
#    Nspike_CA3_exc[k]+=1
#for k in CA3i_i:
#    Nspike_CA3_inh[k]+=1
#for k in CA1e_i:
#    Nspike_CA1_exc[k]+=1
#for k in CA1i_i:
#    Nspike_CA1_inh[k]+=1
#
#figure()
#plot(zeros(10000),Nspike_CA1_exc,'+')
#
#def coherence_fenetre_glissante(raster_i,raster_t,N,fenetre=0.05,wind=3*msecond,numpairs=2000):
#    tmin=0
#    tmax=tmin+fenetre
#    time_vect=[]
#    all_coherence=[]
#    while tmax<2:
#        extrait_i,extrait_t=fen(raster_i,raster_t,tmin,tmax)
#        c=coherence(extrait_i,extrait_t,N,wind=wind,numpairs=numpairs)
#        #print(c)
#        all_coherence.append(c)
#        time_vect.append(tmin)
#        tmin+=fenetre/4
#        tmax=tmin+fenetre
#    return time_vect,all_coherence
#
#figure()
#T,C=coherence_fenetre_glissante(raster_EC_exc_i,raster_EC_exc_t,10000)
#plot(T,C,label='EC')
#T,C=coherence_fenetre_glissante(raster_DG_exc_i,raster_DG_exc_t,10000)
#plot(T,C,label='DG')
#T,C=coherence_fenetre_glissante(raster_CA3_exc_i,raster_CA3_exc_t,1000)
#plot(T,C,label='CA3')
#T,C=coherence_fenetre_glissante(raster_CA1_exc_i,raster_CA1_exc_t,10000)
#plot(T,C,label='CA1')
#title('Coherence exc')
#legend()
#
#figure()
#T,C=coherence_fenetre_glissante(raster_EC_inh_i,raster_EC_inh_t,1000)
#plot(T,C,label='EC')
#T,C=coherence_fenetre_glissante(raster_DG_inh_i,raster_DG_inh_t,100)
#plot(T,C,label='DG')
#T,C=coherence_fenetre_glissante(raster_CA3_inh_i,raster_CA3_inh_t,100)
#plot(T,C,label='CA3')
#T,C=coherence_fenetre_glissante(raster_CA1_inh_i,raster_CA1_inh_t,1000)
#plot(T,C,label='CA1')
#title('Coherence inh')
#legend()
#
#import ast
#def lecture(filename):
#    data_array=[]
#    file=open(filename,'r')
#    for data in file :
#        d=ast.literal_eval(data[:-1])
#        data_array.append(d)
#    file.close()
#    return data_array
#
#T1,C1=coherence_fenetre_glissante(raster_CA3_exc_i,raster_CA3_exc_t,1000)
#T2,C2=coherence_fenetre_glissante(raster_CA1_exc_i,raster_CA1_exc_t,10000)
#T3,C3=coherence_fenetre_glissante(raster_CA3_inh_i,raster_CA3_inh_t,100)
#T4,C4=coherence_fenetre_glissante(raster_CA1_inh_i,raster_CA1_inh_t,1000)
#
#
#simu=lecture('time_series_som_som_1A.txt')[0][int(33.925*1024):int(33.925*1024)+1800]
#simu=array(simu)/max(simu)*0.6
#time=linspace(0,2,1800)
#
#figure()
#plot(time,simu,label='LFP',color='C0')
#plot(T1,C1,label='CA3 E',color='#9164c8',ls=':')
#plot(T2,C2,label='CA1 E',color='#000064',ls=':')
#plot(T3,C3,label='CA3 I',color='#9164c8')
#plot(T4,C4,label='CA1 I',color='#000064')
#legend()
#xlim(1.2,1.65)
#xticks([1.2,1.3,1.4,1.5,1.6,1.7],[0,100,200,300,400,500,600,700])
#xlabel('Time(ms)')
#ylabel('Coherence measure')

record_dt=1/512*second
#record_dt=1/10000*second
L=int(0.8*second/record_dt)+1

print()
raster_mat_EC_exc=zeros((L,10000))
for ind in range(len(raster_EC_exc_t)):
    raster_mat_EC_exc[int(raster_EC_exc_t[ind]/record_dt),raster_EC_exc_i[ind]]=1
raster_mat_EC_inh=zeros((L,1000))
for ind in range(len(raster_EC_inh_t)):
    raster_mat_EC_inh[int(raster_EC_inh_t[ind]/record_dt),raster_EC_inh_i[ind]]=1

raster_mat_DG_exc=zeros((L,10000))
for ind in range(len(raster_DG_exc_t)):
    raster_mat_DG_exc[int(raster_DG_exc_t[ind]/record_dt),raster_DG_exc_i[ind]]=1
raster_mat_DG_inh=zeros((L,100))
for ind in range(len(raster_DG_inh_t)):
    raster_mat_DG_inh[int(raster_DG_inh_t[ind]/record_dt),raster_DG_inh_i[ind]]=1

raster_mat_CA3_exc=zeros((L,1000))
for ind in range(len(raster_CA3_exc_t)):
    raster_mat_CA3_exc[int(raster_CA3_exc_t[ind]/record_dt),raster_CA3_exc_i[ind]]=1
raster_mat_CA3_inh=zeros((L,100))
for ind in range(len(raster_CA3_inh_t)):
    raster_mat_CA3_inh[int(raster_CA3_inh_t[ind]/record_dt),raster_CA3_inh_i[ind]]=1

raster_mat_CA1_exc=zeros((L,10000))
for ind in range(len(raster_CA1_exc_t)):
    raster_mat_CA1_exc[int(raster_CA1_exc_t[ind]/record_dt),raster_CA1_exc_i[ind]]=1
raster_mat_CA1_inh=zeros((L,1000))
for ind in range(len(raster_CA1_inh_t)):
    raster_mat_CA1_inh[int(raster_CA1_inh_t[ind]/record_dt),raster_CA1_inh_i[ind]]=1


figure()
subplot(411)
plot(raster_mat_EC_exc,'r.')
subplot(412)
plot(raster_mat_DG_exc,'r.')
subplot(413)
plot(raster_mat_CA3_exc,'r.')
subplot(414)
plot(raster_mat_CA1_exc,'r.')

depart_electrode=array([-21, 0, 50])
arrivee_electrode=array([15, 0, 50])
len_elec=norm(arrivee_electrode-depart_electrode)
#    print(len_elec*150*umetre)
dir_elec=(arrivee_electrode-depart_electrode)/norm(arrivee_electrode-depart_electrode)
elec=[]
#psi = arccos(dot(depart_electrode[:-1],arrivee_electrode[:-1])/(norm(depart_electrode[:-1])*norm(arrivee_electrode[:-1])))
psi = arccos(dot(dir_elec,array([0,1,0])))
diametre = 400/150 #en fait c'est un rayon
for t in linspace(0,1,33):
    centre=(1-t)*depart_electrode+t*arrivee_electrode
    #print(centre)
    for theta in arange(0,2*pi,pi/6):
        point=[centre[0]+diametre*cos(theta)*cos(psi),centre[1]-diametre*cos(theta)*sin(psi),centre[2]+diametre*sin(theta)]
        #print(point)
        elec.append(point)
elec_array=array(elec[:144]+elec[252:])

t=int(0.5*second/record_dt) #t_debut
L=int(0.8*second/record_dt)

def init_fig(inh=False):
    global pECe,pECi,pDGe,pDGi,pCA3e,pCA3i, pCA1e, pCA1i,texte,ptest,pelec,p_scalex,p_scaley,p_scalez,text_scale,text_EC,text_DG,text_CA3,text_CA1
    ptest=ax3D.scatter(positions_EC_exc[:,0], positions_EC_exc[:,1], positions_EC_exc[:,2], c=[0.6,0.6,0.6], marker ='o')

    pECe=ax3D.scatter(positions_EC_exc[:,0], positions_EC_exc[:,1], positions_EC_exc[:,2], c=[0.6,0.6,0.6], marker ='o')
    if inh :
        pECi=ax3D.scatter(positions_EC_inh[:,0], positions_EC_inh[:,1], positions_EC_inh[:,2], c=[0.2,0.2,0.2], marker ='o')

    pDGe=ax3D.scatter(positions_DG_exc[:,0], positions_DG_exc[:,1], positions_DG_exc[:,2], c=[0.6,0.6,0.6], marker ='o')
    if inh:
        pDGi=ax3D.scatter(positions_DG_inh[:,0], positions_DG_inh[:,1], positions_DG_inh[:,2], c=[0.2,0.2,0.2], marker ='o')

    pCA3e=ax3D.scatter(positions_CA3_exc[:,0], positions_CA3_exc[:,1], positions_CA3_exc[:,2], c=[0.6,0.6,0.6], marker ='o')
    if inh:
        pCA3i=ax3D.scatter(positions_CA3_inh[:,0], positions_CA3_inh[:,1], positions_CA3_inh[:,2], c=[0.2,0.2,0.2], marker ='o')

    pCA1e=ax3D.scatter(positions_CA1_exc[:,0], positions_CA1_exc[:,1], positions_CA1_exc[:,2], c=[0.6,0.6,0.6], marker ='o')
    if inh:
        pCA1i=ax3D.scatter(positions_CA1_inh[:,0], positions_CA1_inh[:,1], positions_CA1_inh[:,2], c=[0.2,0.2,0.2], marker ='o')
    texte=ax3D.text(20,10,120,'time=0',fontsize=10)
    pelec=ax3D.scatter(elec_array[:,0],elec_array[:,1],elec_array[:,2], c=[0,0,0], marker ='o')
    p_scalex=ax3D.plot([-15,-15+1000/150],[-30,-30],[0,0],'k',linewidth=10)
    p_scaley=ax3D.plot([-15,-15],[-30,-30+1000/150],[0,0],'k',linewidth=10)
    p_scalez=ax3D.plot([-15,-15],[-30,-30],[0,0+1000/150],'k',linewidth=10)
    text_scale=ax3D.text(-15,-30,-15,'1000 $\mu$m','y',fontsize=10)
    text_EC=ax3D.text(25,-30,100,'EC','y',fontsize=10)
    text_DG=ax3D.text(25,0,100,'DG','y',fontsize=10)
    text_CA3=ax3D.text(10,20,100,'CA3','y',fontsize=10)
    text_CA1=ax3D.text(-10,20,100,'CA1','y',fontsize=10)

def update_fig(i,inh=False):
    global t
    global pECe,pECi,pDGe,pDGi,pCA3e,pCA3i, pCA1e, pCA1i,texte,ptest,p_scalex,p_scaley,p_scalez,text_scale,text_EC,text_DG,text_CA3,text_CA1
    print(str(int(t-0.5*second/record_dt))+'/'+str(L-int(0.5*second/record_dt)))


    active_DG_exc=where(raster_mat_DG_exc[t,:]==1)[0]
    color_DG_exc=array([[0.4,0.6,0.4] for i in range(10000)])
    color_DG_exc[active_DG_exc,:]=[1,0,0]
#    print(len(active_DG_exc))
#    print(len(where(color_DG_exc==[1,0,0])[0]))
#    pDGe.set_color(color_DG_exc)
    size_DG_exc=array([1 for i in range(10000)])
    size_DG_exc[active_DG_exc]=5
#    pDGe.set_sizes(size_DG_exc)
#    pDGe.changed()
    if inh :
        active_DG_inh=where(raster_mat_DG_inh[t,:]==1)[0]
        color_DG_inh=array([[0.2,0.4,0.2] for i in range(100)])
        color_DG_inh[active_DG_inh,:]=[0,1,0]
    #    pDGi.set_color(color_DG_inh)
        size_DG_inh=array([1 for i in range(1000)])
        size_DG_inh[active_DG_inh]=5
    #    pDGi.set_sizes(size_DG_inh)
    #    pDGi.changed()

    active_CA3_exc=where(raster_mat_CA3_exc[t,:]==1)[0]
    color_CA3_exc=array([[0.4,0.4,0.6] for i in range(1000)])
    color_CA3_exc[active_CA3_exc,:]=[1,0,0]
#    pCA3e.set_color(color_CA3_exc)
    size_CA3_exc=array([1 for i in range(1000)])
    size_CA3_exc[active_CA3_exc]=5
#    pCA3e.set_sizes(size_CA3_exc)
#    pCA3e.changed()
    if inh :
        active_CA3_inh=where(raster_mat_CA3_inh[t,:]==1)[0]
        color_CA3_inh=array([[0.2,0.2,0.4] for i in range(100)])
        color_CA3_inh[active_CA3_inh,:]=[0,1,0]
    #    pCA3i.set_color(color_CA3_inh)
        size_CA3_inh=array([1 for i in range(100)])
        size_CA3_inh[active_CA3_inh]=5
    #    pCA3i.set_sizes(size_CA3_inh)
    #    pCA3i.changed()

    active_CA1_exc=where(raster_mat_CA1_exc[t,:]==1)[0]
    color_CA1_exc=array([[0.6,0.4,0.6] for i in range(10000)])
    color_CA1_exc[active_CA1_exc,:]=[1,0,0]
#    pCA1e.set_color(color_CA1_exc)
    size_CA1_exc=array([1 for i in range(10000)])
    size_CA1_exc[active_CA1_exc]=5
#    pCA1e.set_sizes(size_CA1_exc)
#    pCA1e.changed()
    if inh :
        active_CA1_inh=where(raster_mat_CA1_inh[t,:]==1)[0]
        color_CA1_inh=array([[0.4,0.2,0.4] for i in range(1000)])
        color_CA1_inh[active_CA1_inh,:]=[0,1,0]
    #    pCA1i.set_color(color_CA1_inh)
        size_CA1_inh=array([1 for i in range(1000)])
        size_CA1_inh[active_CA1_inh]=5
    #    pCA1i.set_sizes(size_CA1_inh)
    #    pCA1i.changed()

    active_EC_exc=where(raster_mat_EC_exc[t,:]==1)[0]
#    print(len(active_EC_exc))
    color_EC_exc=array([[0.6,0.4,0.4] for i in range(10000)])
    color_EC_exc[active_EC_exc,:]=[1,0,0]
#    print(len(where(color_EC_exc==[1,0,0])[0]))
#    pECe.set_facecolor(color_EC_exc)
#    print(color_EC_exc[1,:])
    size_EC_exc=array([1 for i in range(10000)])
    size_EC_exc[active_EC_exc]=5
#    pECe.set_sizes(size_EC_exc)
#    pECe.changed()
    if inh :
        active_EC_inh=where(raster_mat_EC_inh[t,:]==1)[0]
        color_EC_inh=array([[0.4,0.2,0.2] for i in range(1000)])
        color_EC_inh[active_EC_inh,:]=[0,1,0]
    #    pECi.set_color(color_EC_inh)
        size_EC_inh=array([1 for i in range(1000)])
        size_EC_inh[active_EC_inh]=5
    #    pECi.set_sizes(size_EC_inh)
    #    pECi.changed()

    fig2.clear()
    ax3D = fig2.add_subplot(111, projection='3d')
    texte=ax3D.text(20,10,150,'time = '+str((int((t*1./512-0.5)*1000)))+' ms',fontsize=10)

    ptest=ax3D.scatter(positions_EC_exc[:,0], positions_EC_exc[:,1], positions_EC_exc[:,2], c=color_EC_exc, marker ='o',s=size_EC_exc)
#    print(color_EC_exc.tolist().count([1,0,0]))
#    print(pECe.get_facecolor(),pECe.get_facecolor().shape)
    if inh :
        pECi=ax3D.scatter(positions_EC_inh[:,0], positions_EC_inh[:,1], positions_EC_inh[:,2], c=color_EC_inh, marker ='o',s=size_EC_inh)

    pCA3e=ax3D.scatter(positions_CA3_exc[:,0], positions_CA3_exc[:,1], positions_CA3_exc[:,2], c=color_CA3_exc, marker ='o',s=size_CA3_exc)
    if inh :
        pCA3i=ax3D.scatter(positions_CA3_inh[:,0], positions_CA3_inh[:,1], positions_CA3_inh[:,2], c=color_CA3_inh, marker ='o',s=size_CA3_inh)

    pDGe=ax3D.scatter(positions_DG_exc[:,0], positions_DG_exc[:,1], positions_DG_exc[:,2], c=color_DG_exc, marker ='o',s=size_DG_exc)
    if inh:
        pDGi=ax3D.scatter(positions_DG_inh[:,0], positions_DG_inh[:,1], positions_DG_inh[:,2], c=color_DG_inh, marker ='o',s=size_DG_inh)
#    print(len(where(color_DG_exc==[1,0,0])[0]))
#    print(pDGe.get_facecolor(),pDGe.get_facecolor().shape)
#    print(len(where(pDGe.get_facecolor()==[1,0,0,1])[0]))


    pCA1e=ax3D.scatter(positions_CA1_exc[:,0], positions_CA1_exc[:,1], positions_CA1_exc[:,2], c=color_CA1_exc, marker ='o',s=size_CA1_exc)
    if inh:
        pCA1i=ax3D.scatter(positions_CA1_inh[:,0], positions_CA1_inh[:,1], positions_CA1_inh[:,2], c=color_CA1_inh, marker ='o',s=size_CA1_inh)

    pECe=ax3D.scatter(positions_EC_exc[:,0], positions_EC_exc[:,1], positions_EC_exc[:,2], c=color_EC_exc, marker ='o',s=size_EC_exc)
#    print(pECe.get_facecolor().tolist().count([1,0,0,1]))
#    print(pECi.get_facecolor().tolist().count([0,1,0,1]))
    pelec=ax3D.scatter(elec_array[:,0],elec_array[:,1],elec_array[:,2], c=[0,0,0], marker ='o')
    p_scalex=ax3D.plot([-15,-15+1000/150],[-30,-30],[0,0],'k',linewidth=10)
    p_scaley=ax3D.plot([-15,-15],[-30,-30+1000/150],[0,0],'k',linewidth=10)
    p_scalez=ax3D.plot([-15,-15],[-30,-30],[0,0+1000/150],'k',linewidth=10)
    text_scale=ax3D.text(-15,-30,-15,'1000 $\mu$m','y',fontsize=10)
    text_EC=ax3D.text(25,-30,100,'EC','y',fontsize=10)
    text_DG=ax3D.text(25,0,100,'DG','y',fontsize=10)
    text_CA3=ax3D.text(10,20,100,'CA3','y',fontsize=10)
    text_CA1=ax3D.text(-10,20,100,'CA1','y',fontsize=10)
    ax3D.azim=(t//2)%360
    ax3D.elev=40
    ax3D.set_xticklabels([])
    ax3D.set_yticklabels([])
    ax3D.set_zticklabels([])

#    fig2.canvas.draw()
#    fig2.canvas.flush_events()
    t+=1
    return

def forceUpdate(event,inh=False):
    global pECe,pECi,pDGe,pDGi,pCA3e,pCA3i, pCA1e, pCA1i
    pECe.changed()
    if inh:
        pECi.changed()
    pDGe.changed()
    if inh:
        pDGi.changed()
    pCA3e.changed()
    if inh:
        pCA3i.changed()
    pCA1e.changed()
    if inh:
        pCA1i.changed()

def redraw(inh=False):
    global pECe,pECi,pDGe,pDGi,pCA3e,pCA3i, pCA1e, pCA1i
    fig2.clear()
    pECe=ax3D.scatter(positions_EC_exc[:,0], positions_EC_exc[:,1], positions_EC_exc[:,2], c=color_EC_exc, marker ='o',markersize=size_EC_exc)
    if inh:
        pECi=ax3D.scatter(positions_EC_inh[:,0], positions_EC_inh[:,1], positions_EC_inh[:,2], c=color_EC_inh, marker ='o',markersize=size_EC_inh)

    pDGe=ax3D.scatter(positions_DG_exc[:,0], positions_DG_exc[:,1], positions_DG_exc[:,2], c=color_DG_exc, marker ='o',markersize=size_DG_exc)
    if inh:
        pDGi=ax3D.scatter(positions_DG_inh[:,0], positions_DG_inh[:,1], positions_DG_inh[:,2], c=color_DG_inh, marker ='o',markersize=size_DG_inh)

    pCA3e=ax3D.scatter(positions_CA3_exc[:,0], positions_CA3_exc[:,1], positions_CA3_exc[:,2], c=color_CA3_exc, marker ='o',markersize=size_CA3_exc)
    if inh:
        pCA3i=ax3D.scatter(positions_CA3_inh[:,0], positions_CA3_inh[:,1], positions_CA3_inh[:,2], c=color_CA3_inh, marker ='o',markersize=size_CA3_inh)

    pCA1e=ax3D.scatter(positions_CA1_exc[:,0], positions_CA1_exc[:,1], positions_CA1_exc[:,2], c=color_CA1_exc, marker ='o',markersize=size_CA1_exc)
    if inh:
        pCA1i=ax3D.scatter(positions_CA1_inh[:,0], positions_CA1_inh[:,1], positions_CA1_inh[:,2], c=color_CA1_inh, marker ='o',markersize=size_CA1_inh)



fig2 = figure()
ax3D = fig2.add_subplot(111, projection='3d')
init_fig()
#fig2.canvas.mpl_connect('draw_event',forceUpdate)
ani = animation.FuncAnimation(fig2, update_fig,frames=L,repeat=False, interval=100)
ani.save('try_animation.gif',writer='imagemagick', fps=512)



def lecture(filename):
    data_array=[]
    file=open(filename,'r')
    for data in file :
        d=ast.literal_eval(data[:-1])
        data_array.append(d)
    file.close()
    return data_array

def init_fig3():
    global pLFP
    time=linspace(0,0.3,int(0.3*second/record_dt))
#    pLFP=ax3.plot(time,LFP,'white')
    pLFP, = ax3.plot([],[])
    xlim(0, 0.3)
    ylim(m*1.05,1.05)
    xlabel('Time (s)')
    ylabel('LFP')

tmax=int(0.8*second/record_dt)
t0=int(0.5*second*1024*Hz) #t_debut
t=0 #t_debut

LFP=lecture(path+'/LFP.txt')[0][t0+50:int(0.8*second*1024*Hz)+50][::2]
LFP=array(LFP)/max(LFP)
m=min(LFP)

def update_fig3(i):
    global t,pLFP
#    print(str(t))
    pLFP.set_data(array(range(t))*record_dt,LFP[:t])
#    ax3.plot([(t-1)*record_dt,t*record_dt,(t+1)*record_dt],[LFP[t-1],LFP[t],LFP[t+1]],'blue')
    t+=1
    return


fig3 = figure(figsize=(5,3))
ax3=fig3.add_subplot(111)
fig3.subplots_adjust(bottom=0.2)
init_fig3()
#fig2.canvas.mpl_connect('draw_event',forceUpdate)
ani = animation.FuncAnimation(fig3, update_fig3,frames=int(1*second/record_dt),repeat=False, interval=100)
ani.save('try_animated_plot.gif',writer='imagemagick', fps=512)




##2D
#
#def init_fig():
#    global pECe,pECi,pDGe,pDGi,pCA3e,pCA3i, pCA1e, pCA1i,texte
#    pECe=ax3D.scatter(positions_EC_exc[:,0], positions_EC_exc[:,1], c=[0.6,0.6,0.6], marker ='o')
#    pECi=ax3D.scatter(positions_EC_inh[:,0], positions_EC_inh[:,1], c=[0.2,0.2,0.2], marker ='o')
#
#    pDGe=ax3D.scatter(positions_DG_exc[:,0], positions_DG_exc[:,1], c=[0.6,0.6,0.6], marker ='o')
#    pDGi=ax3D.scatter(positions_DG_inh[:,0], positions_DG_inh[:,1], c=[0.2,0.2,0.2], marker ='o')
#
#    pCA3e=ax3D.scatter(positions_CA3_exc[:,0], positions_CA3_exc[:,1], c=[0.6,0.6,0.6], marker ='o')
#    pCA3i=ax3D.scatter(positions_CA3_inh[:,0], positions_CA3_inh[:,1], c=[0.2,0.2,0.2], marker ='o')
#
#    pCA1e=ax3D.scatter(positions_CA1_exc[:,0], positions_CA1_exc[:,1], c=[0.6,0.6,0.6], marker ='o')
#    pCA1i=ax3D.scatter(positions_CA1_inh[:,0], positions_CA1_inh[:,1], c=[0.2,0.2,0.2], marker ='o')
#    texte=ax3D.text(20,10,'time=0',fontsize=10)
#
#def update_fig(i):
#    global t
#    print(str(t)+'/'+str(L))
#    active_EC_exc=where(raster_mat_EC_exc[t,:]==1)
#    color_EC_exc=array([[0.2,0.2,0.2] for i in range(10000)])
#    color_EC_exc[active_EC_exc,:]=[1,0,0]
##    print(color_EC_exc[1,:])
#    pECe.set_color(color_EC_exc)
#    pECe.changed()
#    active_EC_inh=where(raster_mat_EC_inh[t,:]==1)
#    color_EC_inh=array([[0.6,0.6,0.6] for i in range(1000)])
#    color_EC_inh[active_EC_inh,:]=[0,1,0]
#    pECi.set_color(color_EC_inh)
#    pECi.changed()
#
#    active_DG_exc=where(raster_mat_DG_exc[t,:]==1)
#    color_DG_exc=array([[0.6,0.6,0.6] for i in range(10000)])
#    color_DG_exc[active_DG_exc,:]=[1,0,0]
#    pDGe.set_color(color_DG_exc)
#    pDGe.changed()
#    active_DG_inh=where(raster_mat_DG_inh[t,:]==1)
#    color_DG_inh=array([[0.2,0.2,0.2] for i in range(100)])
#    color_DG_inh[active_DG_inh,:]=[0,1,0]
#    pDGi.set_color(color_DG_inh)
#    pDGi.changed()
#
#    active_CA3_exc=where(raster_mat_CA3_exc[t,:]==1)
#    color_CA3_exc=array([[0.6,0.6,0.6] for i in range(1000)])
#    color_CA3_exc[active_CA3_exc,:]=[1,0,0]
#    pCA3e.set_color(color_CA3_exc)
#    pCA3e.changed()
#    active_CA3_inh=where(raster_mat_CA3_inh[t,:]==1)
#    color_CA3_inh=array([[0.2,0.2,0.2] for i in range(100)])
#    color_CA3_inh[active_CA3_inh,:]=[0,1,0]
#    pCA3i.set_color(color_CA3_inh)
#    pCA3i.changed()
#
#    active_CA1_exc=where(raster_mat_CA1_exc[t,:]==1)
#    color_CA1_exc=array([[0.6,0.6,0.6] for i in range(10000)])
#    color_CA1_exc[active_CA1_exc,:]=[1,0,0]
#    pCA1e.set_color(color_CA1_exc)
#    pCA1e.changed()
#    active_CA1_inh=where(raster_mat_CA1_inh[t,:]==1)
#    color_CA1_inh=array([[0.2,0.2,0.2] for i in range(1000)])
#    color_CA1_inh[active_CA1_inh,:]=[0,1,0]
#    pCA1i.set_color(color_CA1_inh)
#    pCA1i.changed()
#
#    texte.set_text('time = '+str(t*1000//1024)+' ms')
#
#    t+=1
#    return
#
#def forceUpdate(event):
#    global pECe,pECi,pDGe,pDGi,pCA3e,pCA3i, pCA1e, pCA1i
#    pECe.changed()
#    pECi.changed()
#    pDGe.changed()
#    pDGi.changed()
#    pCA3e.changed()
#    pCA3i.changed()
#    pCA1e.changed()
#    pCA1i.changed()
#
#
#fig2 = figure()
#ax3D = fig2.add_subplot(111)
#init_fig()
##fig2.canvas.mpl_connect('draw_event',forceUpdate)
#ani = animation.FuncAnimation(fig2, update_fig,frames=L,repeat=False, interval=100)
#ani.save('try_animation.gif',writer='imagemagick', fps=512)
