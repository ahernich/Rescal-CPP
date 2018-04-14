/* 
 * rescal.cpp - C++ implementation of RESCAL-ALS, an algorithm for learning 
 * a latent feature representation of a knowledge graph.
 *
 * This code is derived from the original python implementation of RESCAL-ALS 
 * by Maximilian Nickel (https://github.com/mnick/rescal.py).
 * 
 * Copyright (C) 2018 André Hernich
 *
 * This file is part of Rescal C++.
 *
 * Rescal C++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Rescal C++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <chrono>
#include <random>

#include "rescal.h"

using rescal::ALS;
using rescal::Tensor;

using namespace arma;
using namespace std;

namespace {

/**
 * Coordinate (row-column pair) in a matrix.
 */
struct Coordinate 
{
    uword row; ///< Index of a row in a matrix (0 <= row < number of rows)
    uword col; ///< Index of a column in a matrix (0 <= col < number of columns)
};

}

/******************************************************************************
 * Local functions
 *****************************************************************************/

/**
 * Naive computation of the norm of a sparse matrix, without eigenvector 
 * decomposition. Might be slower than Armadillo's norm(), but can be used
 * if Armadillo's norm() fails due to a failure during eigenvalue computation.
 *
 * \param[in] A
 *   Matrix for which we want to compute the norm.
 * \returns
 *   The squared norm of \p A.
 */
static double naiveSquaredNorm(const sp_mat& A) 
{
    long double norm = 0; 
    for (uword a : A)
        norm += a*a;
    return sqrt(norm);
}

/**
 * Computes the sum norm of a tensor. In order to not carry out any unnecessary
 * computation, we define the sum norm of a tensor X as the sum of the squared 
 * norms of X's slices (rather than the norms of X's slices).
 *
 * \param[in] tensor
 *   Tensor for which we want to compute the norm.
 * \returns
 *   The norm of \p tensor.
 */
static double computeSumNorm(const Tensor& tensor)
{
    double sumNorm = 0;
    for (const auto& slice : tensor) 
        sumNorm += naiveSquaredNorm(slice);
    return sumNorm;
}

/**
 * Computes the number of matrix elements that come before a given coordinate y,
 * starting from a coordinate x. We assume column-row order of the coordinates.
 *
 * \param[in] x
 *   First coordinate.
 * \param[in] y
 *   Second coordinate. This coordinate is assumed to come before x (including
 *   the possibility that x == y). The function does not check this.
 * \param[in] rows
 *   Number of rows of the matrix.
 * \returns
 *   The number of coordinates that come before \p y in a matrix with \p cols 
 *   rows, starting from coordinate \p x.
 * \see Coordinate
 */
static size_t dist(const Coordinate& x, const Coordinate& y, uword rows)
{
    if (x.col == y.col) 
        return y.row - x.row;
    else 
        return (rows - x.row) + (y.col - x.col - 1) * rows + y.row;
}

/**
 * Shifts a matrix coordinate by a given number of positions in ascending 
 * column-row order.
 *
 * \param[in, out] coord
 *   Coordinate to be shifted. Upon return, \p coord contains the return of 
 *   shifting the original coordinate.
 * \param[in] shift
 *   The total number of positions by which c should be shifted.
 * \param[in] rows
 *   The number of rows of the matrix.
 */
static void shiftCoordinate(Coordinate& coord, size_t shift, uword rows)
{
    size_t targetPos {coord.col * rows + coord.row + shift};
    coord.row = targetPos % rows;
    coord.col = targetPos / rows;
}

/**
 * Takes a sparse matrix \f$A\f$ and an index \f$p\f$ as input and returns the 
 * coordinate of \f$A\f$ in column-row order that corresponds to the \f$p\f$th
 * zero entry in \f$A\f$.
 *
 * \param[in] A
 *   The matrix.
 * \param[in] p
 *   Index of the zero entry in \p A whose coordinate we want to return.
 * \returns
 *   The coordinate of the \p p th zero in \p A.
 */
static Coordinate translate(const sp_mat& A, size_t p)
{
    Coordinate cur {}; // current candidate coordinate
    for (auto it = A.begin(); it != A.end(); ++it)
    {
        size_t d {dist(cur, {it.row(), it.col()}, A.n_rows)};
        if (p < d)
        {
            shiftCoordinate(cur, p, A.n_rows);
            p = 0;
            break;
        }
        else
        {
            // Shift cur to coordinate directly after {it.row(), it.col()}
            shiftCoordinate(cur, d+1, A.n_rows);
            p -= d;
        }
    }
    shiftCoordinate(cur, p, A.n_rows);
    return cur;
}

/**
 * Samples n coordinates at which the given sparse matrix is zero. The 
 * coordinates are sampled uniformly at random from among all coordinates at
 * which the given matrix is zero. The function does not guarantee that the
 * returned coordinates are distinct.
 *
 * \param[in] A
 *   Matrix from which to sample.
 * \param[in] n
 *   Number of samples.
 * \returns
 *   A vector containing exactly n coordinates at which A is zero.
 */
static vector<Coordinate> sampleZeros(const sp_mat& A, size_t n) 
{
    vector<Coordinate> samples;
    const size_t n_zeros {A.n_rows * A.n_cols - A.n_nonzero}; // #zero positions
    if (n_zeros)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine gen(seed);
        uniform_int_distribution<uword> distr(0, n_zeros-1);
        for (size_t i = 0; i != n; ++i)
            samples.push_back(translate(A, distr(gen)));
    }
    return samples;
}

/**
 * Naive computation of the value of the matrix product A*B*C at a given 
 * coordinate.
 *
 * \param[in] A, B, C
 *   The three matrices.
 * \param[in] coord
 *   The coordinate of the matrix A*B*C we are interested in.
 * \returns
 *   The entry at coordinate \p coord in the matrix \p A * \p B * \p C.
 */
static double productAt(const mat& A, 
                        const mat& B, 
                        const mat& C, 
                        const Coordinate& coord)
{
    long double prod = 0;
    for (uword i = 0; i != B.n_rows; ++i)
        for (uword j = 0; j != C.n_rows; ++j)
            prod += A(coord.row, i) * B(i, j) * C(j, coord.col);
    return prod;
}

/******************************************************************************
 * Class member functions
 *****************************************************************************/

ALS::ALS(const Tensor& inputTensor, size_t nFeatures)
    : inputTensor {inputTensor},
      sumNorm {computeSumNorm(inputTensor)},
      nRelations {inputTensor.size()},
      nEntities {(nRelations > 0) ? inputTensor[0].n_rows : 0},
      nFeatures {nFeatures}
{
    // Check that each frontal slice of the input tensor is a square matrix
    // and that all matrices have the same number of rows.
    for (const auto& slice : inputTensor) 
    {
        if (slice.n_rows != nEntities)
            throw std::invalid_argument("The frontal slices of the tensor " 
                                        "do not have the same number of rows.");
        if (slice.n_cols != nEntities)
            throw std::invalid_argument("The frontal slices of the tensor "
                                        "are not square matrices.");
    }

    // Set default parameters
    initMethod = InitMethod::NVECS;
    maxIter = 500;
    fitComputeMethod = FitComputeMethod::EXACT;
    conv = 1e-10;

    // Initialize A and R
    A = mat(nEntities, nFeatures);
    for (size_t i = 0; i != nRelations; ++i)
        Rs.push_back(mat(nFeatures, nFeatures));
}

void ALS::compute(double lambdaA, double lambdaR)
{
    // Initialize A and Rs
    initA();
    updateR(lambdaR);

    // Compute factorization
    double fit = 1;
    for (unsigned i = 0; i != maxIter; ++i) 
    {
        if (fitComputeMethod != FitComputeMethod::NONE) 
        {
            double newFit = computeFit();
            if (abs(fit - newFit) < conv) 
                break;
            fit = newFit;
        }
        updateA(lambdaA);
        updateR(lambdaR);
    }
}

const mat& ALS::getA() const
{
    return A;
}

const vector<mat>& ALS::getRs() const 
{
    return Rs;
}

void ALS::setFactorMatrixInitMethod(InitMethod initMethod)
{
    this->initMethod = initMethod;
}

void ALS::setMaxIterations(unsigned maxIter)
{
    this->maxIter = maxIter;
}

void ALS::enableExactFitComputation(double conv)
{
    fitComputeMethod = FitComputeMethod::EXACT;
    this->conv = conv;
}

void ALS::enableApproximateFitComputation(double conv, size_t nSamples)
{
    fitComputeMethod = FitComputeMethod::APPROX;
    this->conv = conv;
    this->nSamples = nSamples;
}

void ALS::disableFitComputation()
{
    fitComputeMethod = FitComputeMethod::NONE;
}

double ALS::computeFit() const
{
    switch (fitComputeMethod)
    {
    case FitComputeMethod::EXACT:
        return computeExactFit();
    case FitComputeMethod::APPROX:
        return computeApproximateFit();
    case FitComputeMethod::NONE:
        return 0;
    }
}

double ALS::computeExactFit() const
{
    const sp_mat sparse_A {A};
    const sp_mat sparse_At {A.t()};

    double f = 0;
    for (size_t i = 0; i != nRelations; ++i) 
    {
        // Compute current approximation X = A * Rs[i] * A^T of inputTensor[i] 
        // and add the distance between X and inputTensor[i] to f
        sp_mat X {sparse_A * Rs[i] * sparse_At};
        X -= inputTensor[i];
        f += naiveSquaredNorm(X);
    }

    return 1 - f/sumNorm;
}

double ALS::computeApproximateFit() const
{
    mat At {A.t()};

    double f = 0;
    for (size_t i = 0; i != nRelations; ++i)
    {
        sp_mat X(nEntities, nEntities);

        // Insert into X all entries of A * Rs[i] * A.t() at coordinates where
        // inputTensor[i] != 0
        for (auto it = inputTensor[i].begin(); it != inputTensor[i].end(); ++it)
            X(it.row(), it.col()) = 
                productAt(A, Rs[i], At, {it.row(), it.col()});

        // Fill X with nSamples samples of A * Rs[i] * A.t() at coordinates
        // where inputTensor[i] = 0
        for (const Coordinate& c : sampleZeros(inputTensor[i], nSamples))
            X(c.row, c.col) = productAt(A, Rs[i], At, c);

        // Add norm of inputTensor[i] - X to f
        X -= inputTensor[i];
        f += naiveSquaredNorm(X);
    }

    return 1 - f/sumNorm;
}

void ALS::initA() 
{
    switch (initMethod) 
    {
    case InitMethod::NVECS:
        {   
            // To compute eigenvectors using eigs_sym, we first need to compute
            // a symmetric matrix S from inputTensor...
            sp_mat S(nEntities, nEntities);
            for (const auto& X : inputTensor)
                S += X + X.t();

            // Now initialize the i-th column of A to be the i-th largest 
            // eigenvector of S (eigval is only needed to call eigs_sym()).
            vec eigval;
            eigs_sym(eigval, A, S, nFeatures);
        }
        break;
    case InitMethod::RANDOM:
        A.randu(nEntities, nFeatures);
        break;
    }
}

void ALS::updateA(double lambdaA) 
{
    mat E(nEntities, nFeatures, fill::zeros);
    mat F(nFeatures, nFeatures, fill::zeros);

    const mat AtA {A.t()*A};

    for (size_t i = 0; i != nRelations; ++i)
    {
        const mat R {Rs[i]};
        const mat Rt {R.t()};

        E += inputTensor[i]*A*Rt + inputTensor[i].t()*A*R;
        F += R*AtA*Rt + Rt*AtA*R;
    }

    F += lambdaA * eye<mat>(nFeatures, nFeatures);

    // Avoid computing A = E * inv(F) directly. Instead, use faster and more
    // accurate method using solve() (see Armadillo documentation).
    solve(A, F.t(), E.t());
    inplace_trans(A);
}

void ALS::updateR(double lambdaR) 
{
    vec s;
    mat U, V;
    svd_econ(U, s, V, A);

    const mat Ut {U.t()};
    const mat Vt {V.t()};

    mat D {kron(s, s)};
    D.for_each( [lambdaR](double& val) { val /= val*val + lambdaR; } );
    D.reshape(nFeatures, nFeatures);

    for (size_t i = 0; i != nRelations; ++i) 
    {
        const mat Z {D % (Ut * inputTensor[i] * U)};
        Rs[i] = V * Z * Vt;
    }
}
