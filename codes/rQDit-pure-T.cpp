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

void pauli_matrices();

void make_H(double A,double B);

void make_V();

void unitary_evolve();

double simulate();

double uniform();

double normal(double mean, double std);

int poisson(double lambda);

////////////////////// end functions



////////////////////// variables

const int D=50;

const double P0=1.0/D;

const double pi=4*atan(1.0);

const double epsilon=2;

const double delta=0.5;

const double r=0;

int  L;

Matr H = Matr::Zero(D, D);

Matr V = Matr::Zero(D, D);

Stat psi = Stat::Zero(D);

Stat phi = Stat::Zero(D);

Stat chi = Stat::Zero(D);

vector<Matr> Paulis(4);

////////////////////// end variables



////////////////////// main

int main() {


    double Ps,p1,p2;
    double T1,T2;

    pauli_matrices();

    T1=0;
    T2=0;
    int Nsample=1;
    for(int sample=0;sample<Nsample;sample++){

        L=0;
        Ps=0;
        while(Ps<1-delta){
            L+=1;
            initial();
            Ps=simulate();
        }
        T1+=L;
        T2+=L*L;


        ofstream output("Tpy-e2r0.dat",ios::out| ios::app);
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


//////////////////////// noisy H

void make_V() {

    double ws=epsilon/L;

    //V=ws*(Paulis[0]+Paulis[0].adjoint());
    V=ws*(Paulis[1]+Paulis[1].adjoint());
    //V=ws*(Paulis[2]+Paulis[2].adjoint());



}


//////////////////////// initial

void initial() {

    psi(0)=sqrt(P0);
    for(int a=1;a<D;a++)psi(a)=sqrt((1-P0)/(D-1));

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

    H(a,b)=A*(I-Pab)+B*(I-F)-r*psi(a)*conj(psi(b))+V(a,b);
    H(b,a)=conj(H(a,b));

    }
    }


}

//////////////////////// Time evolution: |ψ(t)⟩ = exp(-iH) |ψ(0)⟩

void unitary_evolve() {

    Matr U = (-1i * H).exp();
    phi = U * psi;

}

//////////////////////// simulate

double simulate() {

        int l;
        double Pr,Ps,Pmax;
        double A,B,dP;

        l=0;
        dP=1;
        Pmax=0;
        while((l<L) && (dP>0)){

            make_V();

            Ps=0;
            for(int la=1;la<11;la++){
            for(int lb=1;lb<11;lb++){
                A=la*(2*pi)/(11-1);
                B=lb*(2*pi)/(11-1);

                make_H(A,B);
                unitary_evolve();
                Pr=real(phi(0)*conj(phi(0)));
                if(Pr>Ps){
                    Ps=Pr;
                    chi=phi;
                }

            }
            }
            psi=chi;
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
