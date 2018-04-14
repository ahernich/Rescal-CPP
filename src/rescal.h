/* 
 * rescal.h - Interface to RESCAL-ALS, an algorithm for learning a latent 
 * feature representation of a knowledge graph.
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

#ifndef RESCAL_H
#define RESCAL_H

#include <vector>

#include <armadillo>

/**
 * Namespace containing the main interface, ALS, to RESCAL-ALS and the Tensor
 * datatype that is required to provide a knowledge graph as input to ALS.
 *
 * \see Tensor
 * \see ALS
 */
namespace rescal {

/** 
 * A tensor. The tensor is represented as the sequence of its slices. The 
 * slices should have the same dimension.
 */
using Tensor = std::vector<arma::sp_mat>;

/**
 * This class implements the RESCAL-ALS algorithm for learning a latent feature
 * representation of a knowledge graph.
 *
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
 * 
 *  - Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel.
 *    A Three-Way Model for Collective Learning on Multi-Relational Data.
 *    ICML 2011, Bellevue, WA, USA
 *  
 *  - Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel.
 *    Factorizing YAGO: Scalable Machine Learning for Linked Data.
 *    WWW 2012, Lyon, France
 *
 * The following example illustrates how to use ALS:
 * \code
 * arma::sp_mat X1(n, n);
 * // Initialize X1
 *
 * // ...
 *
 * arma::sp_mat Xm(n, n);
 * // Initialize Xm
 *
 * Tensor X {X1, ..., Xm};
 *
 * // Run RESCAL-ALS to learn a representation of the knowledge graph in terms
 * // of r latent features. More precisely, run RESCAL-ALS to factor X into a 
 * // matrix A and matrices R[0], ..., R[m-1], where A has dimension n x r and 
 * // the R[i] have dimension r x r.
 * ALS als(X, r);
 * als.compute();
 *
 * // Retrieve result
 * const arma::mat& A = als.getA();
 * const std::vector<arma::mat>& Rs = als.getRs();
 *
 * std::cout << "Factor matrix A:\n" << A << "\n\n";
 * for (int i = 0; i != m; ++i) 
 * {
 *   std::cout << "Factor matrix R[" << i << "]:\n" << Rs[i] << "\n\n";
 *   std::cout << "A*R*A^T = \n" << (A * Rs[i] * A.t()) << "\n\n";
 * }
 * \endcode 
 *
 * The main function, compute(), has several parameters to fine-tune RESCAL-ALS,
 * including parameters for the maximum number of iterations and the method of
 * initializing the factor matrix A. See the documentation of compute() for 
 * details.
 */
class ALS
{
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
     * Initializes ALS. Use compute() to run RESCAL-ALS in order to compute the
     * tensor factorization.
     *
     * \param[in] inputTensor
     *   The tensor representation of a knowledge graph whose latent feature
     *   representation we want to learn. 
     * \param[in] nFeatures
     *   The number of latent features to learn (also known as the rank of the 
     *   tensor factorization).
     *
     * \throws invalid_argument
     *   If \p inputTensor does not consist of square matrices of the same 
     *   dimension.
     *
     * \see ALS for information on how to represent a knowledge graph 
     *   as a tensor.
     */
    ALS(const Tensor& inputTensor, size_t nFeatures);

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
     * There are various parameters that influence the performance of compute():
     *
     *   - The method of initializing the factor matrix \f$A\f$. The default is 
     *     InitMethod::NVECS. Use setFactorMatrixInitMethod() to set the method
     *     used by the next call to compute().
     *
     *   - The maximum number of iterations. The default value is 500. Use 
     *     setMaxIterations() to set the maximum number of iterations used by 
     *     the next call to compute().
     *
     *   - The method of computing the fit value (a measure of how well the 
     *     current tensor factorization approximates the input tensor) after 
     *     each iteration. The default is to compute an exact fit value based
     *     on the \f$\ell2\f$ distance between the input tensor and the tensor 
     *     with slices \f$A R_i A^T\f$. For large inputs, it is recommended 
     *     to use an approximation of this fit value or turn off computation
     *     of fit values entirely. Use enableExactFitComputation(), 
     *     enableApproximateFitComputation() or disableFitComputation() to set
     *     the method used by the next call to compute().
     * 
     * \param[in] lambdaA
     *   Regularization parameter for the computation of the factor matrix 
     *   \f$A\f$. See the papers on RESCAL for a description of this parameter.
     * \param[in] lambdaR
     *   Regularization parameter for the computation of the factor matrices 
     *   \f$R_i\f$. See the papers on RESCAL for a description of this
     *   parameter.
     *
     * \throws std::runtime_error
     *   If there is an error initializing the factor matrix \f$A\f$. This can 
     *   only happen if InitMethod::NVECS is used as the method of initializing 
     *   \f$A\f$.
     *
     * \see getA(), getRs(),
     * \see setFactorMatrixInitMethod()
     * \see setMaxIterations(), 
     * \see enableExactFitComputation(), enableApproximateFitComputation(), 
     *   disableFitComputation()
     */
    void compute(double lambdaA = 0, double lambdaR = 0);

    /**
     * Returns the matrix \f$A\f$ of the RESCAL factorization derived by 
     * compute(). 
     *
     * \see compute()
     */
    const arma::mat& getA() const;

    /**
     * Returns the matrices \f$R_i\f$ of the RESCAL factorization derived by 
     * compute(). 
     *
     * \returns 
     *   A vector whose i-th component is the matrix \f$R_i\f$ of the RESCAL 
     *   factorization.
     * \see compute()
     */
    const std::vector<arma::mat>& getRs() const;

    /**
     * Sets the method of initializing the factor matrix \f$A\f$ at the 
     * beginning of compute().
     *
     * \param[in] initMethod
     *   Method of initializing the factor matrix \f$A\f$ for compute().
     * \see compute()
     */
    void setFactorMatrixInitMethod(InitMethod initMethod);

    /**
     * Sets the maximum number of iterations for compute(). 
     *
     * \param[in] maxIter
     *   The new maximum number of iterations.
     * \see compute()
     */
    void setMaxIterations(unsigned maxIter);

    /**
     * Enables exact computation of fit values. See computeFit() or compute()
     * for an explanation of fit values.
     *
     * \param[in] conv
     *   Threshold for convergence. If the difference between the fit values
     *   of consecutive iterations is less than conv, then compute() will stop.
     *
     * \see computeFit(), compute()
     * \see enableApproximateFitComputation(), disableFitComputation()
     */
    void enableExactFitComputation(double conv);

    /**
     * Enables approximate computation of fit values. See computeFit() or 
     * compute() for an explanation of fit values. 
     *
     * The fit value computed is based on the non-zero entries of the input 
     * tensor and a number of zero positions of the input tensor. The zero 
     * positions are selected uniformly at random from among all the zero 
     * positions of the input tensor.
     *
     * \param[in] conv
     *   Threshold for convergence. If the difference between the fit values
     *   of consecutive iterations is less than conv, then compute() will stop.
     * \param[in] nSamples 
     *   Number of samples to be taken from the zero positions of the input 
     *   tensor.
     *
     * \see computeFit(), compute()
     * \see enableExactFitComputation(), disableFitComputation()
     */
    void enableApproximateFitComputation(double conv, size_t nSamples);

    /**
     * Disables computation of fit values. See computeFit() or compute()
     * for an explanation of fit values.
     *
     * \see computeFit(), compute()
     * \see enableExactFitComputation(), enableApproximateFitComputation()
     */
    void disableFitComputation();

    /**
     * Computes a fit value for the current factor matrices \f$A\f$, \f$R_i\f$, 
     * depending on the current method of computing fit values. 
     *
     * The fit value is a double between 0 and 1. A value of 1 means that
     * \f$A R_i A^T\f$ coincides with the \f$i\f$th slice of the input tensor, 
     * and a value of 0 means that \f$A R_i A^T\f$ is "as far away as possible"
     * from that slice. The closer the fit value is to 1, the closer is each of
     * the matrices \f$A R_i A^T\f$ to the \f$i\f$th slice of the input tensor. 
     *
     * If fit computation is disabled using disableFitComputation(), then the 
     * value 0 will be returned.
     *
     * \see compute()
     * \see enableExactFitComputation()
     * \see enableApproximateFitComputation()
     * \see disableFitComputation()
     */
    double computeFit() const;

private:

    /**
     * Tensor representation of the knowledge graph.
     */
    const Tensor inputTensor;

    /**
     * Sum of squared norms of matrices in input tensor.
     */ 
    const double sumNorm;

    /**
     * Number of relations in the knowledge graph.
     */
    const std::size_t nRelations;

    /**
     * Number of entities in the knowledge graph.
     */
    const std::size_t nEntities;

    /**
     * Number of latent features to learn (rank of the tensor factorization).
     */
    const std::size_t nFeatures;

    /**
     * Method of initializing the factor matrix A at the beginning of compute().
     */
    InitMethod initMethod;

    /**
     * Maximum number of iterations for compute().
     */
    std::size_t maxIter;

    /**
     * Constants for specifying the method of computing the fit value after
     * each iteration of RESCAL.
     *
     * \see compute()
     */
    enum class FitComputeMethod 
    {
        EXACT,  ///< Exact, but slow. Not recommended for large inputs.
        APPROX, ///< Fairly fast, but not too accurate. Good for testing.
        NONE    ///< No computation of fit values. Perform all iterations.
    };

    /**
     * Method of computing a fit value after an iteration of compute().
     */
    FitComputeMethod fitComputeMethod;

    /**
     * Number of samples to take from the zero positions of inputTensor, if 
     * \p fitComputeMethod == FitComputeMethod::APPROX.
     */
    size_t nSamples;

    /**
     * If \p fitComputeMethod is EXACT or APPROX, \p conv is the threshold for 
     * the fit value of the current factorization compared to the input tensor
     * where the algorithm will stop.
     */
    double conv;

    /**
     * Matrix representing the entities in the knowledge base in terms of 
     * latent features. This matrix has dimension nEntities x nFeatures.
     */
    arma::mat A;

    /**
     * Vector containing exactly nRelations matrices of dimension 
     * nFeatures x nFeatures. The i-th matrix represents the relationships
     * between latent features with respect to relation i.
     */
    std::vector<arma::mat> Rs;

    /**
     * Initialize the factor matrix A.
     */
    void initA();

    /**
     * Update the factor matrix A during an interation of RESCAL-ALS.
     *
     * \param[in] lambdaA
     *   Regularization parameter.
     */
    void updateA(double lambdaA);

    /**
     * Updates the factor matrices in _R during an interation of RESCAL-ALS.
     *
     * \param[in] lambdaR
     *   Regularization parameter.
     */
    void updateR(double lambdaR);

    /**
     * Computes the fit value for the current factor matrices A and R[i]. This
     * is a value between 0 and 1. A value of 1 means 
     * \f[
     *   \texttt{A} \texttt{R[i]} \texttt{A}^T = \texttt{inputTensor[i]}
     * \f]
     * and a value of 0 means that \f$\texttt{A} \texttt{R[i]} \texttt{A}^T\f$ 
     * is as far away as possible from inputTensor[i]. The closer the fit value
     * is to 1, the closer is each of the matrices 
     * \f$\texttt{A} \texttt{R[i]} \texttt{A}^T\f$ to inputTensor[i].
     */
    double computeExactFit() const;

    /**
     * Computes an approximate fit value. This is based on the non-zero entries
     * in the input tensor and a number of zero positions of the input tensor. 
     * The zero positions are selected uniformly at random from among all the
     * zero positions of the input tensor. The number of samples is controlled
     * by the member variable \p nSamples.
     */
    double computeApproximateFit() const;
};

} // end namespace rescal

#endif // RESCAL_H