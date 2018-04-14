# Rescal C++

Rescal C++ is a C++ implementation of RESCAL-ALS, 
an algorithm for learning a latent feature representation of a knowledge graph.
This implementation is derived from the original
[Python implementation of Rescal](https://github.com/mnick/rescal.py) 
by Maximilian Nickel.

## Compilation

Rescal C++ requires:

* a C++ compiler supporting the C++ 11 standard or higher;
* [Armadillo](http://arma.sourceforge.net) >= 8.500.

To compile, 
you will need to tell your compiler to use the C++ 11 standard
and to link with libarmadillo.
You may also want to enable optimization.
For example:
```
g++ -c -std=c++11 -O2 -I<path to armadillo headers> rescal.cpp
... compile other sources ...
g++ rescal.o <other object files> -larmadillo
```

## RESCAL-ALS

For a detailed description of RESCAL-ALS and its applications, see: 

1. Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel.
   *A Three-Way Model for Collective Learning on Multi-Relational Data.*
   ICML 2011, Bellevue, WA, USA

2. Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel.
   *Factorizing YAGO: Scalable Machine Learning for Linked Data.*
   WWW 2012, Lyon, France

RESCAL-ALS expects a knowledge graph to be represented as a tensor. For the purposes of RESCAL-ALS, we can think of a tensor as a sequence \f$X_1,X_2,\ldots\f$ of matrices, all with the same number of rows and columns. The matrix \f$X_k\f$ is called the \f$k\f$th slice of the tensor. 
If the knowledge graph has \f$n\f$ entities and \f$m\f$ relations,
then the tensor has \f$m\f$ slices of dimension \f$n \times n\f$, one for each relation. The \f$k\f$th slice \f$X_k\f$ is the adjacency matrix of the \f$k\f$th relation. Thus, \f$X_k[i,j] = 1\f$ if entity \f$i\f$ is related to entity \f$j\f$ in relation \f$k\f$, and \f$X_k[i,j] = 0\f$ otherwise.

Given the tensor representation \f$X_1,\ldots,X_m\f$ of a knowledge graph with \f$n\f$ entities and an integer \f$r\f$, RESCAL-ALS computes an \f$n \times r\f$ matrix \f$A\f$ and \f$r \times r\f$ matrices \f$R_1,\ldots,R_m\f$ such that 
\f[
    X_i \approx A R_i A^T.
\f]
The matrix \f$A\f$ represents the entities of the knowledge base in terms of \f$r\f$ latent features, whereas \f$R_i\f$ represents the relationships between these features with respect to relation \f$i\f$.
   
## Usage

The class rescal::ALS provides the main interface to RESCAL-ALS. 
The library uses the matrix datatypes (mat, sp_mat) 
from [Armadillo](http://arma.sourceforge.net);
please consult the excellent documentation to learn more about 
how to work with matrices in Armadillo.

The following example runs RESCAL-ALS 
on the tensor representation \f$X_1,\ldots,X_m\f$ of a knowledge graph
and outputs the factor matrices \f$A\f$ and \f$R_i\f$,
along with the approximations \f$A R_i A^T\f$ of \f$X_i\f$:
```c++
#include <armadillo>
#include <iostream>
#include <rescal.h>

using namespace rescal;

int main()
{
  // Prepare input tensor X for knowledge graph with n entities and m relations

  arma::sp_mat X1(n, n);
  // Initialize X1

  // ...

  arma::sp_mat Xm(n, n);
  // Initialize Xm

  Tensor X {X1, ..., Xm};

  // Run RESCAL-ALS to learn a representation of the knowledge graph in terms
  // of r latent features. More precisely, run RESCAL-ALS to factor X into a 
  // matrix A and matrices R[0], ..., R[m-1], where A has dimension n x r and 
  // the R[i] have dimension r x r.
  ALS als(X, r);
  als.compute();

  // Retrieve result
  const arma::mat& A = als.getA();
  const std::vector<arma::mat>& Rs = als.getRs();

  std::cout << "Factor matrix A:\n" << A << "\n\n";
  for (int i = 0; i != m; ++i) 
  {
    std::cout << "Factor matrix R[" << i << "]:\n" << Rs[i] << "\n\n";
    std::cout << "A*R*A^T = \n" << (A * Rs[i] * A.t()) << "\n\n";
  }
}
```
After constructing the tensor representation \f$X\f$ of the knowledge graph,
we instantiate Rescal::ALS 
by providing \f$X\f$ and the number \f$r\f$ of latent features to learn. 
Then, we call the method Rescal::ALS::compute() to run RESCAL-ALS on \f$X\f$.
The method has several parameters to fine-tune RESCAL-ALS,
including parameters for the maximum number of iterations
and the initialization method for the factor matrix \f$A\f$. 
See the documentation of Rescal::ALS::compute() for the default values. 

## Author

André Hernich

## License

Rescal C++ is licensed under the GPLv3 http://www.gnu.org/licenses/gpl-3.0.txt
