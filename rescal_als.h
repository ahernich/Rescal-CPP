/* 
 * rescal_als.h - C++ implementation of RESCAL-ALS, an algorithm for learning 
 * a latent feature representation of a knowledge graph.
 *
 * The code is based on the original python implementation of RESCAL-ALS by 
 * Maximilian Nickel (https://github.com/mnick/rescal.py).
 * 
 * Copyright (C) 2018 André Hernich
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __RESCAL_ALS_H__
#define __RESCAL_ALS_H__

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>

#include <unsupported/Eigen/KroneckerProduct>

#include <SymEigsSolver.h>          // from Spectra library
#include <MatOp/SparseGenMatProd.h> // from Spectra library

namespace Rescal { 

/// \defgroup types Tensors, Matrices, Vectors, and Arrays
/// 
/// Classes for tensors, matrices, vectors, and arrays based on Eigen's matrix
/// and array classes (Eigen::SparseMatrix, Eigen::Matrix, and Eigen::Array).
///
/// \see Eigen (https://eigen.tuxfamily.org/) for details on Eigen::SparseMatrix,
///     Eigen::Matrix, and Eigen::Array.
///
/// @{ 

/**
 * SparseMatrix type imported from Eigen.
 */
template<typename Scalar>
using SparseMatrix = Eigen::SparseMatrix<Scalar>;

/** 
 * A tensor with coefficients of type \p Scalar. The tensor is represented as 
 * the sequence of its slices. The slices should have the same dimension.
 *
 * \tparam Scalar
 *   Type of the coefficients of the matrix.
 */
template<typename Scalar>
using Tensor = std::vector<SparseMatrix<Scalar>>;

/**
 * A matrix with coefficients of type \p Scalar.
 *
 * \tparam Scalar
 *   Type of the coefficients of the matrix.
 */
template<typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * A vector with coefficients of type \p Scalar.
 *
 * \tparam Scalar
 *   Type of the coefficients of the vector.
 */
template<typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/**
 * An array with coefficients of type \p Scalar.
 *
 * \tparam Scalar
 *   Type of the coefficients of the array.
 */
template<typename Scalar>
using Array = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// @} END OF DOXYGEN GROUP types

/**
 * Computes the norm of the given tensor. Here, we define the norm as the sum 
 * of the squared norms of the slices.
 *
 * \param[in] tensor
 *   Tensor for which we want to compute the norm.
 * \returns
 *   The norm of \p tensor.
 */
template<typename Scalar>
double computeSumNorm(const Tensor<Scalar>& tensor)
{
    double n = 0;
    for (const SparseMatrix<Scalar>& slice : tensor)
        n += slice.squaredNorm();
    return n;
}

/**
 * Compute the k eigenvectors with largest magnitude for a given symmetric
 * sparse matrix.
 *
 * \param[in] A
 *   Symmetric sparse matrix.
 * \param[in] k
 *   The number of eigenvectors to compute. This should be a positive number
 *   smaller than the number of rows of A.
 * \returns
 *   A matrix with \p A.cols() rows and \p k columns. Column i is the i-th 
 *   largest eigenvector of A with respect to magnitude.
 * 
 * \throws std::out_of_range
 *   If k is not a positive integer smaller than the number of rows of A.
 * \throws std::runtime_error
 *   If there is an error during the computation of the eigenvectors.
 */
template<typename Scalar>
Matrix<Scalar> eigenvectors(const SparseMatrix<Scalar>& A, int k) 
{
    if (k < 1 || k >= A.rows())
        throw std::out_of_range("k should be an integer in [1,A.rows()-1]");
    Spectra::SparseGenMatProd<Scalar> op(A);
    long ncv = std::min<long>(A.rows(), std::max(2*k+1, 20));
    Spectra::SymEigsSolver<Scalar, 
                           Spectra::LARGEST_MAGN, 
                           Spectra::SparseGenMatProd<Scalar>> solv(&op, k, ncv);
    solv.init();
    solv.compute(1000, 1e-10, Spectra::LARGEST_MAGN);
    if (solv.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("Unable to compute eigenvectors.");
    return solv.eigenvectors();
}

/**
 * This class implements the RESCAL-ALS algorithm for learning a latent feature
 * representation of a knowledge graph.
 *
 * RESCAL-ALS

 * RESCAL-ALS expects a knowledge graph to be given as the sequence of the 
 * adjacency matrices of its relations. More precisely, assume that the 
 * knowledge graph contains entitites \f$E_1,\ldots,E_n\f$ and relations 
 * \f$R_1,\ldots,R_m\f$. We then represent each relation \f$R_i\f$ by its 
 * adjacency matrix \f$X_i\f$, that is, as an \f$n \times n\f$ matrix whose 
 * entry in row \f$r\f$ and column \f$c\f$ is 1 if the knowledge graph contains
 * a \f$R_i\f$-labeled edge from \f$E_r\f$ to \f$E_c\f$, and 0 otherwise. The 
 * matrices \f$X_1,\ldots,X_m\f$ form a tensor. The datatype for representing 
 * tensors is Tensor. 
 *
 * For a detailed description of RESCAL-ALS and its applications, see: 
 * <ol>
 *   <li>
 *     Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel.
 *     <i>A Three-Way Model for Collective Learning on Multi-Relational Data.</i>
 *     ICML 2011, Bellevue, WA, USA
 *   </li>
 *   <li>
 *     Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel.
 *     <i>Factorizing YAGO: Scalable Machine Learning for Linked Data.</i>
 *     WWW 2012, Lyon, France
 *   </li>
 * </ol>
 *
 * The following is a typical use of ALS:
 * \code
 * // Prepare input tensor X for knowledge graph with n entities and m relations
 * 
 * Rescal::SparseMatrix<double> X1(n, n);
 * ...
 * 
 * ...
 *
 * Rescal::SparseMatrix<double> Xm(n, n);
 * ...
 *
 * Rescal::Tensor<double> X {X1,...,Xm};
 *
 * // Run RESCAL-ALS to learn a representation of the knowledge graph in terms
 * // of k latent features. More precisely, run RESCAL-ALS to factor X into a 
 * // matrix A and matrices R[0], ..., R[m-1], where A has dimension n x k and 
 * // the R[i] have dimension k x k.
 * Rescal::ALS<double> als(X, k);
 * als.compute();
 *
 * const Rescal::Matrix<double>& A = als.getA();
 * const std::vector<Rescal::Matrix<double>>& R = als.getRs();
 * \endcode 
 *
 * \tparam Scalar
 *   Type of matrix coefficients (i.e., coefficients of the slices of the input
 *   tensor and of the factor matrices).
 *
 * \todo Optimise code
 * \todo Add support for attribute matrices.
 */
template<typename Scalar=double>
class ALS
{
    /**
     * Tensor representation of the knowledge graph.
     */
    const Tensor<Scalar>& _inputTensor;

    /**
     * Number of entities in the knowledge graph.
     */
    const Eigen::Index _nEntities;

    /**
     * Number of relations in the knowledge graph.
     */
    const std::size_t _nRelations;

    /**
     * Number of latent features to learn (rank of the tensor factorization).
     */
    const std::size_t _nFeatures;

    /**
     * Sum of squared norms of matrices in input tensor.
     */ 
    const double _sumNorm;

    /**
     * Matrix representing the entities in the knowledge base in terms of 
     * latent features. This matrix has dimension _nEntities x _nFeatures.
     */
    Matrix<Scalar> _A;

    /**
     * Vector containing exactly _nRelations matrices of dimension 
     * _nFeatures x _nFeatures. The i-th matrix represents the relationships
     * between latent features with respect to relation i.
     */
    std::vector<Matrix<Scalar>> _R;

public:

    /**
     * Constants for specifying the method of initializing the factor matrix A.
     *
     * \see compute()
     */
    enum class InitMethod
    {
        NVECS, ///< Initializes A based on the eigenvectors of the input tensor.
        RANDOM ///< Use random values to initialize A.
    };

    /**
     * Initializes ALS.
     *
     * \param[in] inputTensor
     *   The tensor representation of a knowledge graph whose latent feature
     *   representation we want to learn. 
     * \param[in] nFeatures
     *   The number of latent features to learn (also known as the rank of the 
     *   tensor factorization).
     * \throws invalid_argument
     *   If \p inputTensor does not consist of square matrices of the same 
     *   dimension.
     *
     * \see ALS for information on how to represent a knowledge graph 
     *   as a tensor.
     */
    ALS(const Tensor<Scalar>& inputTensor, std::size_t nFeatures)
        : _inputTensor {inputTensor}, 
          _nEntities {inputTensor.size() > 0 ? inputTensor[0].rows() : 0},
          _nRelations {inputTensor.size()}, 
          _nFeatures {nFeatures},
          _sumNorm {computeSumNorm(inputTensor)}
    {
        // Check that each frontal slice of the input tensor is a square matrix
        // and that all matrices have the same number of rows.
        for (const SparseMatrix<Scalar>& slice : _inputTensor) 
        {
            if (slice.rows() != _nEntities)
                throw std::invalid_argument("The frontal slices of the tensor " 
                                            "do not have the same number "
                                            "of rows.");
            if (slice.cols() != _nEntities)
                throw std::invalid_argument("The frontal slices of the tensor "
                                            "are not square matrices.");
        }

        // Initialize _A and _R
        _A = Matrix<Scalar>::Zero(_nEntities, _nFeatures);
        for (std::size_t i = 0; i < _nRelations; i++)
            _R.push_back(Matrix<Scalar>::Zero(_nFeatures, _nFeatures));
    }

    /**
     * Runs the RESCAL-ALS algorithm. 
     *
     * This method factors the input tensor into a matrix \f$A\f$, representing
     * the entities in terms of latent features, and matrices \f$R_i\f$ (one
     * for each relation), representing relationships between latent features 
     * with respect to relation \f$i\f$. The goal is that the \f$i\f$th slice
     * of the input tensor is approximated well by the matrix product
     * \f[
     *     A R_i A^T.
     * \f]
     * The matrices \f$A\f$ and \f$R_i\f$ can be obtained using getA() and 
     * getRs().
     * 
     * \param[in] maxIter
     *   The maximum number of iterations. The method will return after at most
     *   this number of iterations.
     * \param[in] useFit
     *   If \c true, the method computes the fit of the factorization compared 
     *   to the input tensor and stops if the difference between consecutive 
     *   fit values is less than \p conv (or the maximum number of iterations 
     *   is reached). For large input tensors, it is advisable to use \c false.
     * \param[in] conv
     *   Threshold for the fit of the factorization compared to the input tensor
     *   where the algorithm will stop, provided \p useFit is \c true.
     * \param[in] initMethod
     *   Method of initializing the factor matrix A.
     * \param[in] lambdaA
     *   Regularization parameter for factor matrix A.
     * \param[in] lambdaR
     *   Regularization parameter for factor matrices R_k.
     * \throws std::runtime_error
     *   If there is an error initializing the factor matrix A. This can only
     *   happen if \p initMethod is InitMethod::NVECS.
     *
     * \see getA(), getRs()
     */
    void compute(int maxIter = 500, 
                 bool useFit = true,
                 double conv = 1e-5, 
                 InitMethod initMethod = InitMethod::NVECS,
                 double lambdaA = 0, 
                 double lambdaR = 0)
    {
        // Initialize A
        if (initMethod == InitMethod::NVECS) 
        {
            SparseMatrix<Scalar> S(_nEntities, _nEntities);
            for (std::size_t i = 0; i < _nRelations; i++) 
            {
                S += _inputTensor[i];
                S += SparseMatrix<Scalar>(_inputTensor[i].transpose());
            }
            _A = eigenvectors(S, _nFeatures);
        }
        else // InitMethod::RANDOM
        {
            _A = Matrix<Scalar>::Random(_nEntities, _nFeatures);
        }

        // Initialize R
        _updateR(lambdaR);

        // Compute factorization
        double fit = 1;
        for (int itr = 0; itr < maxIter; ++itr) 
        {
            if (useFit) 
            {
                double newFit = _computeFit();
                if (abs(fit - newFit) < conv) 
                    break;
                fit = newFit;
            }

            _updateA(lambdaA);
            _updateR(lambdaR);
        }
    }

    /**
     * Return the matrix representing the entities in the knowledge base 
     * in terms of latent features.
     *
     * The method assumes that compute() was executed. 
     */
    inline const Matrix<Scalar>& getA() const
    {
        return _A;
    }

    /**
     * Return a vector whose i-th component is a matrix representing the
     * relationships between latent features with respect to relation i.
     *
     * The method assumes that compute() was executed. 
     */
    inline const std::vector<Matrix<Scalar>>& getRs() const 
    {
        return _R;
    }

private:

    /**
     * Update the factor matrix _A during an interation of RESCAL-ALS.
     *
     * \param[in] lambdaA
     *   Regularization parameter.
     */
    void _updateA(double lambdaA)
    {
        Matrix<Scalar> E = Matrix<Scalar>::Zero(_nEntities, _nFeatures);
        Matrix<Scalar> F = Matrix<Scalar>::Zero(_nFeatures, _nFeatures);

        const Matrix<Scalar> AtA = _A.transpose() * _A;

        for (std::size_t i = 0; i < _nRelations; i++) 
        {
            E += _inputTensor[i] * _A * _R[i].transpose() + 
                 _inputTensor[i].transpose() * _A * _R[i];
            F += _R[i] * AtA * _R[i].transpose() + 
                 _R[i].transpose() * AtA * _R[i];
        }

        // Regularization 
        const Matrix<Scalar> I = 
            lambdaA * Matrix<Scalar>::Identity(_nFeatures, _nFeatures);

        _A = E * (F + I).inverse();
    }

    /**
     * Updates the factor matrices in _R during an interation of RESCAL-ALS.
     *
     * \param[in] lambdaR
     *   Regularization parameter.
     */
    void _updateR(double lambdaR)
    {
        const Eigen::BDCSVD<Matrix<Scalar>> 
            svd(_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Vector<Scalar>& S = svd.singularValues();
        const Matrix<Scalar>& U = svd.matrixU();
        const Matrix<Scalar>& V = svd.matrixV();

        Array<Scalar> D {Eigen::kroneckerProduct(S, S).eval()};
        D = D / (D*D + lambdaR);
        D = Eigen::Map<Array<Scalar>>(D.data(), _nFeatures, _nFeatures);

        for (std::size_t i = 0; i < _nRelations; i++) 
        {
            const Array<Scalar> Z 
                {D * (U.transpose() * _inputTensor[i] * U).array()};
            _R[i] = V * Z.matrix() * V.transpose();
        }
    }

    /**
     * Computes the fit value for the current factor matrices _A and _R[i]. The
     * closer the fit value is to 1, the closer is each of the matrices 
     * \f$\texttt{\_A} \texttt{\_R[i]} \texttt{\_A}^T\f$ to _inputTensor[i].
     */
    double _computeFit() const
    {
        double f = 0;
        const SparseMatrix<Scalar> AS {_A.sparseView()};
        for (std::size_t i = 0; i < _nRelations; i++) 
        {
            const SparseMatrix<Scalar> RS {_R[i].sparseView()};
            const SparseMatrix<Scalar> approx {AS * RS * AS.transpose()};
            f += (_inputTensor[i] - approx).squaredNorm();
        }
        return 1 - f/_sumNorm;
    }

};

} // end namespace Rescal

#endif // __RESCAL_ALS_H__