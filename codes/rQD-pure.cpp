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

void make_H(double t);

void make_H_noisy();

void pauli_string();

void pauli_matrices();

void unitary_evolve();

double uniform();

double normal(double mean, double std);

int poisson(double lambda);

////////////////////// end functions



////////////////////// variables

const int  N=8;
const int  D=(1<<N);
const int  L=50;

const double P0=1.0/D;

const double r=1.0;

const double epsilon=1.0;

const double pi=4*atan(1.0);


Matr H= Matr::Zero(D, D);

Stat psi= Stat::Zero(D);

vector<Matr> Paulis(4);


////////////////////// end variables



////////////////////// main

int main() {


    pauli_matrices();


    double t,alpha,Ps;
    double p1,p2;
    double P1[L],P2[L];
    for(int l=0;l<L;l++){
        P1[l]=0;
        P2[l]=0;
    }

    int Nsample=1;
    for(int sample=0;sample<Nsample;sample++){

        initial();

        for(int l=0;l<L;l++){

            t=l/double(L-1);
            alpha=atan(sqrt((1-P0)/P0));
            t=(1-sqrt(P0/(1-P0))*tan((1-2*t)*alpha))/2;

            make_H_noisy();

            make_H(t);

            unitary_evolve();

            Ps=real(psi(0)*conj(psi(0)));

            P1[l]+=Ps;
            P2[l]+=Ps*Ps;
        }

        ofstream output("P.dat");
        for(int l=0;l<L;l++){
            p1=P1[l]/(sample+1);
            p2=P2[l]/(sample+1);
            p2=sqrt((p2-p1*p1)/sample);
            output<<l<<" "<<p1<<" "<<p2<<" "<<sample+1<<endl;
        }
        output.close();

    }


    return 0;
}



////////////////////////////////////////////////////////////////////////



/////////////////////// Kronecker (tensor) product of matrices

Matr kron(const Matr &A, const Matr &B) {

    Matr C(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    return C;

}




/////////////////////// Generate Pauli matrices (X, Y, Z, I)

void pauli_matrices() {

    Paulis[0] = (Matr(2, 2) << 0, 1, 1, 0).finished();      // X
    Paulis[1] = (Matr(2, 2) << 0, -1i, 1i, 0).finished();   // Y
    Paulis[2] = (Matr(2, 2) << 1, 0, 0, -1).finished();     // Z
    Paulis[3] = Matr::Identity(2, 2);                       // I

}



/////////////////////// Build N-qubit Pauli operator (e.g., "XZI")

Matr pauli_string(const string &s) {

    Matr result = Matr::Identity(1, 1);
    for (char c : s) {
        switch (c) {
            case 'X': result = kron(result, Paulis[0]); break;
            case 'Y': result = kron(result, Paulis[1]); break;
            case 'Z': result = kron(result, Paulis[2]); break;
            case 'I': result = kron(result, Paulis[3]); break;
            default: cerr << "Invalid Pauli operator!" << endl;
        }
    }
    return result;

}




/////////////////////// noisy H

void make_H_noisy() {

    double ws;

    H = Matr::Zero(D, D);

    // Apply each Pauli string with probability
    for (int i = 0; i < N; ++i) {
    for (int k = 0; k < 3; ++k) {
        string s(N, 'I');
        s[i] = "XYZ"[k];
        Matr P = pauli_string(s);
        ws=normal(0,epsilon/L);
        H +=ws * P;
    }
    }


}



/////////////////////// initial

void initial() {

    psi(0)=sqrt(P0);
    for(int a=1;a<D;a++)psi(a)=sqrt((1-P0)/(D-1));

}


/////////////////////// Hamiltonian

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

    H(a,b)=(1-t)*(I-Pab)+t*(I-F)-r*psi(a)*conj(psi(b))+H(a,b);
    H(b,a)=conj(H(a,b));

    }
    }


}



/////////////////////// Time evolution: |ψ(t)⟩ = exp(-iH) |ψ(0)⟩

void unitary_evolve() {

    Matr U = (-1i * H).exp();
    psi = U * psi;

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
