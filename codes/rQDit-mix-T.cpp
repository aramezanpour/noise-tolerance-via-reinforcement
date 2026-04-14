#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions> // For matrix exponential

using namespace Eigen;
using namespace std;

typedef MatrixXcd Matr; // Complex matrix type
typedef VectorXcd Stat;  // Wave function type



////////////////////// functions


void alloc_mem();

void initial();

void make_H(double A,double B);

void pauli_matrices();

void unitary_evolve();

void pauli_channel();

double simulate();

double uniform();

double normal(double mean, double std);

int poisson(double lambda);


////////////////////// end functions



////////////////////// variables

const int  D=100;

const double P0=1.0/D;

const double pi=4*atan(1.0);

const double epsilon=4.0;

const double delta=0.5;

int  L;

double r;

double Pe[3];

Matr H= Matr::Zero(D, D);

Matr rho= Matr::Zero(D, D);

Matr phi= Matr::Zero(D, D);

Matr chi= Matr::Zero(D, D);

vector<Matr> Paulis(4);


////////////////////// end variables



////////////////////// main

int main() {


    pauli_matrices();


    double Ps,p1,p2;
    double T1,T2;

    T1=0;
    T2=0;
    int Nsample=1;
    for(int sample=0;sample<Nsample;sample++){

        L=0;
        Ps=0;
        while(Ps<1-delta){
            L+=1;
            Ps=simulate();
        }

        T1+=L;
        T2+=L*L;

        //cout<<sample<<" "<<L<<endl;

        ofstream output("T.dat",ios::out| ios::app);
            p1=T1/(sample+1);
            p2=T2/(sample+1);
            p2=sqrt((p2-p1*p1)/sample);
            output<<D<<" "<<p1<<" "<<p2<<" "<<sample+1<<endl;
        output.close();


    }



    return 0;
}



////////////////////////////////////////////////////////////////////////



//////////////////////// Generate Pauli matrices (X, Y, Z, I)

void pauli_matrices() {

    complex<double> I(0,1);

    Paulis[0] = Matr::Zero(D, D);      // X
    for(int a=0;a<D-1;a++){
        Paulis[0](a+1,a)=1;
    }
    Paulis[0](0,D-1)=1;


    Paulis[2] = Matr::Zero(D, D);      // Z
    for(int a=0;a<D;a++)Paulis[2](a,a)=exp(I*2.0*pi*double(a)/double(D));


    Paulis[1] = Paulis[0]*Paulis[2];   // Y
    Paulis[3] = Matr::Identity(D, D);  // I

}



//////////////////////// initial

void initial() {


    for(int a=0;a<D;a++){
    for(int b=0;b<a+1;b++){

        double Pab=0;
        if(a==b){
            if(a==0){
            Pab=P0;
            }else{
            Pab=(1-P0)/(D-1);
            }
        }else{
            if(a==0 || b==0){
            Pab=sqrt(P0*(1-P0)/(D-1));
            }else{
            Pab=(1-P0)/(D-1);
            }
        }
        rho(a,b)=Pab;
        rho(b,a)=Pab;

    }
    }


}

//////////////////////// Hamiltonian

void make_H(double A,double B) {


    int I,F;
    double Pab;


    for(int a=0;a<D;a++){
    for(int b=0;b<a+1;b++){

    I=0;
    if(a==b)I=1;
    F=0;
    if(a==b && a==0)F=1;

    Pab=0;
    if(a==b){
      if(a==0){
          Pab=P0;
      }else{
          Pab=(1-P0)/(D-1);
      }
    }else{
      if(a==0 || b==0){
          Pab=sqrt(P0*(1-P0)/(D-1));
      }else{
          Pab=(1-P0)/(D-1);
      }
    }

    H(a,b)=A*(I-Pab)+B*(I-F)-r*rho(a,b);
    H(b,a)=conj(H(a,b));

    }
    }


}

//////////////////////// Unitary evolution: ρ → e^{-iHt} ρ e^{iHt}

void unitary_evolve() {

    Matr U = (-1i * H).exp();
    phi = U * rho * U.adjoint();

}

//////////////////////// Pauli channel with error probabilities px, py, pz

void pauli_channel() {


    Matr phi_new = (1 - epsilon/L) * phi;
    for (int k=0; k<3; k++){
        phi_new += (epsilon/L) * Pe[k] * (Paulis[k] * phi * Paulis[k].adjoint());
    }
    phi=phi_new;


}

///////////////////////// simulation

double simulate(){

        double Pr,Ps,Pmax,dP;
        double A,B;
        int l;

        initial();

        l=0;
        dP=1;
        Pmax=0;
        while((l<L)&&(dP>0)){

            Pe[0]=1;
            Pe[1]=0;
            Pe[2]=1;
            double z=Pe[0]+Pe[1]+Pe[2];
            Pe[0] = Pe[0]/z;
            Pe[1] = Pe[1]/z;
            Pe[2] = Pe[2]/z;

            Ps=0;
            for(int lr=1;lr<2;lr++){
            for(int la=1;la<11;la++){
            for(int lb=1;lb<11;lb++){
                r=lr*2;
                A=la*(2.0*pi)/(11-1);
                B=lb*(2.0*pi)/(11-1);

                make_H(A,B);
                unitary_evolve();
                pauli_channel();

                Pr=real(phi(0,0));
                if(Pr>Ps){
                    Ps=Pr;
                    chi=phi;
                }

            }
            }
            }
            rho=chi;
            if(Ps<Pmax)dP=-1;
            Pmax=max(Pmax,Ps);

            l+=1;
        }


        return Pmax;

}


//////////////////////// Uniform (0,1)

double uniform() {

    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);

}


//////////////////////// Normal (mean=0, std=1)

double normal(double mean = 0.0, double std = 1.0) {

    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<double> dist;
    dist = normal_distribution<double>(mean, std);
    return dist(gen);

}


//////////////////////// Poisson (λ)

int poisson(double lambda = 1.0) {

    static random_device rd;
    static mt19937 gen(rd());
    static poisson_distribution<int> dist;
    dist = poisson_distribution<int>(lambda);
    return dist(gen);

}
