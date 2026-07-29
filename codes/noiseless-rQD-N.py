from numpy import *


####

def compute_h(a,b,A,B):

    h0=(A+B-r)/2
    hx=-A*sqrt(P0*(1-P0))-r*real(a*conjugate(b))
    hy=r*imag(a*conjugate(b))
    hz=(A-B-2*A*P0-r*(2*abs(a)*abs(a)-1))/2

    return(h0,hx,hy,hz)

####

def update(a,b):

   pmax=0
   for A in linspace(0,2*pi,10):
     for B in linspace(0,2*pi,10):

            h0,hx,hy,hz=compute_h(a,b,A,B)
            h=sqrt(hx*hx+hy*hy+hz*hz)

            C0=cos(h)
            C1=0
            C2=0
            C3=0
            if(abs(h)>1e-6):
                C1=(hx+1j*hy)*sin(h)/h
                C2=(hx-1j*hy)*sin(h)/h
                C3=hz*sin(h)/h

            aa=exp(-1j*h0)*((C0-1j*C3)*a-C1*b)
            bb=exp(-1j*h0)*(C2*a+(C0+1j*C3)*b)

            p=abs(aa)*abs(aa)
            if(p>pmax):
                pmax=p
                Amax=A
                Bmax=B
                amax=aa
                bmax=bb

   return(amax,bmax)


#### main


r=1
delta=1e-6

output=open('Tr-1e6.dat','w')
output.close()

N=2
while(N<1001):

    D=(1<<N)
    L=D

    P0=1/D
    a=sqrt(P0)
    b=sqrt(1-P0)


    ####

    l=1
    Pmax=0
    Tmax=0
    ts=0
    while(l<L and ts<1):

        a,b=update(a,b)

        P=abs(a)*abs(a)
        if(P>1-delta):
            ts+=1
            Pmax=P
            Tmax=l

        l+=1

    ####

    #print(N,Tmax,Pmax)

    output=open('Tr-1e6.dat','a')
    output.write(str(N)+' '+str(Tmax)+' '+str(Pmax)+'\n')
    output.close()

    N+=1
