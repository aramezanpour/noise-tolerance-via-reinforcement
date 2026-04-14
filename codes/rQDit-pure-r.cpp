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

void pauli_matrices();

void initial();

void make_H(double t);

void make_V();

void unitary_evolve();

double simulate();

double uniform();

double normal(double mean, double std);

int poisson(double lambda);

////////////////////// end functions



////////////////////// variables

const int  D=100;

const int  L=10;

const double P0=1.0/D;

double r;

const double pi=4*atan(1.0);

const double epsilon=4;

Matr H= Matr::Zero(D, D);

Matr V = Matr::Zero(D, D);

Stat psi= Stat::Zero(D);

vector<Matr> Paulis(4);


////////////////////// end variables



////////////////////// main

int main() {


    srand(time(NULL));

    pauli_matrices();


    int M=100;
    double Ps,p1,p2;
    double R[M],P1[M],P2[M];

    for(int lr=0;lr<M;lr++){
        R[lr]=-1+lr*0.1;
        P1[lr]=0;
        P2[lr]=0;
    }

    int Nsample=1;
    for(int sample=0;sample<Nsample;sample++){


        for(int lr=0;lr<M;lr++){
            r=R[lr];
            initial();
            Ps=simulate();
            P1[lr]+=Ps;
            P2[lr]+=Ps*Ps;
        }

        ofstream output("R.dat");
        for(int lr=0;lr<M;lr++){
            p1=P1[lr]/(sample+1);
            p2=P2[lr]/(sample+1);
            p2=sqrt((p2-p1*p1)/sample);
            output<<R[lr]<<" "<<p1<<" "<<p2<<" "<<sample+1<<endl;
        }
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
    //V=ws*(Paulis[1]+Paulis[1].adjoint());
    V=ws*(Paulis[2]+Paulis[2].adjoint());



}


//////////////////////// initial

void initial() {

    psi(0)=sqrt(P0);
    for(int a=1;a<D;a++)psi(a)=sqrt((1-P0)/(D-1));

}



//////////////////////// Hamiltonian

void make_H(double t) {

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

    H(a,b)=(1-t)*(I-Pab)+t*(I-F)-r*psi(a)*conj(psi(b))+V(a,b);
    H(b,a)=conj(H(a,b));

    }
    }


}



//////////////////////// Time evolution: |ψ(t)⟩ = exp(-iH) |ψ(0)⟩

void unitary_evolve() {

    Matr U = (-1i * H).exp();
    psi = U * psi;

}



//////////////////////// simulate

double simulate() {

        double t,alpha,Ps;

        for(int l=0;l<L;l++){

            t=l/double(L-1);
            alpha=atan(sqrt((1-P0)/P0));
            t=(1-sqrt(P0/(1-P0))*tan((1-2*t)*alpha))/2;

            make_V();

            make_H(t);

            unitary_evolve();

        }

        Ps=real(psi(0)*conj(psi(0)));

        return Ps;

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
