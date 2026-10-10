// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/eigensparse.h"

#ifdef CERES_USE_EIGEN_SPARSE

#include <memory>
#include <sstream>
#include "Eigen/OrderingMethods"
#include "Eigen/SparseCore"
#include "SupernodalLDLT.h"
#ifdef CERES_USE_OPENMP
#include "SupernodalLUExecutorOpenMP.h"
#endif
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/linear_solver.h"

namespace ceres {
namespace internal {

// openMVG: the EIGEN_SPARSE backend is implemented with the supernodal LDL^T
// factorization of DirectLUSolvers (third_party/DirectLUSolvers) instead of
// Eigen::SimplicialLDLT. The matrices ceres hands over (normal equations and
// Schur complements) are symmetric positive semi-definite, possibly with a
// Levenberg-Marquardt diagonal; Bunch-Kaufman pivoting keeps the
// factorization well defined if they are not quite definite.

#ifdef CERES_USE_OPENMP
typedef Eigen::supernodal_lu::OpenMPExecutor SupernodalExecutor;
#else
typedef Eigen::supernodal_lu::SerialExecutor SupernodalExecutor;
#endif

template <typename Solver>
class EigenSparseCholeskyTemplate : public EigenSparseCholesky {
 public:
  explicit EigenSparseCholeskyTemplate(int num_threads) : analyzed_(false) {
#ifdef CERES_USE_OPENMP
    solver_.executor() = SupernodalExecutor(num_threads > 0 ? num_threads : 1);
#else
    (void)num_threads;
#endif
  }
  virtual ~EigenSparseCholeskyTemplate() {}
  virtual CompressedRowSparseMatrix::StorageType StorageType() const {
    return CompressedRowSparseMatrix::LOWER_TRIANGULAR;
  }

  virtual LinearSolverTerminationType Factorize(
      const Eigen::SparseMatrix<double>& lhs, std::string* message) {
    if (!analyzed_) {
      solver_.analyzePattern(lhs);

      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Symbolic Analysis: " << solver_.supernodeCount()
                << " supernodes, " << solver_.predictedFactorNonzeros()
                << " factor nonzeros";
      }

      if (solver_.info() != Eigen::Success) {
        *message = "DirectLUSolvers failure. Unable to find symbolic "
                   "factorization: " + solver_.lastErrorMessage();
        return LINEAR_SOLVER_FATAL_ERROR;
      }

      analyzed_ = true;
    }

    solver_.factorize(lhs);
    if (solver_.info() != Eigen::Success) {
      *message = "DirectLUSolvers failure. Unable to find numeric "
                 "factorization: " + solver_.lastErrorMessage();
      return LINEAR_SOLVER_FAILURE;
    }
    return LINEAR_SOLVER_SUCCESS;
  }

  virtual LinearSolverTerminationType Solve(const double* rhs,
                                            double* solution,
                                            std::string* message) {
    CHECK(analyzed_) << "Solve called without a call to Factorize first.";

    // rhs and solution may alias; the solver reads b after it starts
    // writing x (iterative refinement), so take a copy of the rhs.
    const Vector b = ConstVectorRef(rhs, solver_.cols());
    VectorRef(solution, solver_.cols()) = solver_.solve(b);
    if (solver_.info() != Eigen::Success) {
      *message = "DirectLUSolvers failure. Unable to do triangular solve: " +
                 solver_.lastErrorMessage();
      return LINEAR_SOLVER_FAILURE;
    }
    return LINEAR_SOLVER_SUCCESS;
  }

  virtual LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                                std::string* message) {
    CHECK_EQ(lhs->storage_type(), StorageType());
    // A lower triangular CRS matrix is an upper triangular CCS matrix.
    Eigen::MappedSparseMatrix<double, Eigen::ColMajor> eigen_lhs(
        lhs->num_rows(),
        lhs->num_rows(),
        lhs->num_nonzeros(),
        lhs->mutable_rows(),
        lhs->mutable_cols(),
        lhs->mutable_values());
    return Factorize(eigen_lhs, message);
  }

 private:
  bool analyzed_;
  Solver solver_;
};

EigenSparseCholesky* EigenSparseCholesky::Create(
    const OrderingType ordering_type, int num_threads) {
  typedef Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>,
                                Eigen::Upper,
                                Eigen::AMDOrdering<int>,
                                SupernodalExecutor>
      WithAMDOrdering;
  typedef Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>,
                                Eigen::Upper,
                                Eigen::NaturalOrdering<int>,
                                SupernodalExecutor>
      WithNaturalOrdering;
  if (ordering_type == AMD) {
    return new EigenSparseCholeskyTemplate<WithAMDOrdering>(num_threads);
  } else {
    return new EigenSparseCholeskyTemplate<WithNaturalOrdering>(num_threads);
  }
}

EigenSparseCholesky::~EigenSparseCholesky() {}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_USE_EIGEN_SPARSE
