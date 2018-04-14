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
   
RESCAL-ALS expects a knowledge graph to be represented as a tensor. 
For the purposes of RESCAL-ALS, 
we can think of a tensor as a sequence X[1],X[2],... of matrices, 
all with the same number of rows and columns. 
The matrix X[k] is called the k-th slice of the tensor. 
If the knowledge graph has n entities and m relations,
then the tensor has m slices of dimension n x n, one for each relation. 
The k-th slice X[k] is the adjacency matrix of the k-th relation. 
Thus, X[k][i,j] = 1 if entity i is related to entity j in relation k, 
and X[k][i,j] = 0 otherwise.

Given the tensor representation X[1],...,X[m] 
of a knowledge graph with n entities and an integer r, 
RESCAL-ALS computes an n x r matrix A and r x r matrices R[1],...,R[m]
such that X[k] is approximated well by A R[k] A^T.

The matrix A represents the entities of the knowledge graph
in terms of r latent features, 
whereas R[k] represents the relationships between these features 
with respect to relation k.
   
## Usage

The class rescal::ALS provides the main interface to RESCAL-ALS. 
The library uses the matrix datatypes (mat, sp_mat) 
from [Armadillo](http://arma.sourceforge.net);
please consult the excellent documentation to learn more about 
how to work with matrices in Armadillo.

The following example runs RESCAL-ALS 
on the tensor representation X[1],...,X[m] of a knowledge graph
and outputs the factor matrices A and R[k],
along with the approximations A R[k] A^T of the matrix X[k]:
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
After constructing the tensor representation X of the knowledge graph,
we instantiate rescal::ALS 
by providing X and the number r of latent features to learn. 
Then, we call the method rescal::ALS::compute() to run RESCAL-ALS on X.
The method has several parameters to fine-tune RESCAL-ALS,
including parameters for the maximum number of iterations
and the initialization method for the factor matrix A. 
See the documentation of rescal::ALS::compute() for the default values. 

## Author

André Hernich

## License

Rescal C++ is licensed under the GPLv3 http://www.gnu.org/licenses/gpl-3.0.txt
