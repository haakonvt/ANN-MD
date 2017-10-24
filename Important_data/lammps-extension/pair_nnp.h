/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nnp, PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "pair.h"

#include <armadillo>    // Excellent lin.alg. library
#include <vector>

using namespace std;
using namespace arma;
namespace LAMMPS_NS{

class PairNNP : public Pair {
    public:
        PairNNP(class LAMMPS *);
        virtual ~PairNNP();
        virtual void compute(int, int);
        virtual void settings(int, char **);
        void coeff(int, char **);
        double init_one(int, int);
        void init_style();

        // ANN evaluation and backpropagation
        double neural_network(vec);
        mat backprop();

        // Activation functions, with derivatives
        mat act_func_sigmoid(mat);
        mat ddx_act_func_sigmoid(mat);
        mat act_func_tanh(mat);
        mat ddx_act_func_tanh(mat);

        // Cutoff function(s), with derivatives
        mat    cut_func_cos(mat, double);
        double cut_func_cos_single(double, double);
        mat    ddR_cut_func_cos(mat, double);
        double ddR_cut_func_cos_single(double, double);
        mat    cut_func_tanh(mat, double);
        double cut_func_tanh_single(double, double);
        mat    ddR_cut_func_tanh(mat, double);
        double ddR_cut_func_tanh_single(double, double);

        /* Selection of symmetry functions: G2 and G5
           with corresponding derivatives */
        double G2(double, double, double, double);
        double G5(double, double, double, double, double, double, double);
        void ddR_G2(mat, double, double, double, mat&);
        void ddR_G5(double, double, double, double, double, double, double, double,
                    double, double, double, double, double, double *, double *);

    protected:
        int i,nelements;
        double cutoff, pot_eng;

        int N; // Number of hidden layers.
        int M; // Number of nodes per hidden layer (always constant)

        /* Lists of matrices/vectors, using the layer-based matrix description
           of ANNs in thesis. */
        int nmbr_of_ANN_inputs, nmbr_of_ANN_outputs;
        vector<vector<double>> list_sf_params;
        vector<mat> list_of_weights, list_of_weights_T, list_of_biases,
                    summed_inputs, node_outputs;

        void allocate();
        void read_file(char *);
        // virtual void setup_params();
};
}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style Stillinger-Weber requires atom IDs

This is a requirement to use the SW potential.

E: Pair style Stillinger-Weber requires newton pair on

See the newton command.  This is a restriction to use the SW
potential.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open Stillinger-Weber potential file %s

The specified SW potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in Stillinger-Weber potential file

Incorrect number of words per line in the potential file.

E: Illegal Stillinger-Weber parameter

One or more of the coefficients defined in the potential file is
invalid.

E: Potential file has duplicate entry

The potential file has more than one entry for the same element.

E: Potential file is missing an entry

The potential file does not have a needed entry.

*/
