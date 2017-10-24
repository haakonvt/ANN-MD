/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Yongnan Xiong (HNU), xyn@hnu.edu.cn
                         Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#define _USE_MATH_DEFINES // May be necessary to use M_PI from math.h
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_nnp.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

#include <fstream>  // To read ANN and symm.vec. from file

using namespace std;
using namespace arma;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNNP::PairNNP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0; // Not supported since we have angle dependenies
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1; // Not only a pair style since energies are computed from more than one neighbor
  nelements = 1;     // Number of unique elements
  cutoff = 0;        // Read from input file
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNP::~PairNNP()
{
  if (copymode) return;

  if (allocated){
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

double PairNNP::neural_network(vec input_symm_vec){
    // Forward evaluation of the trained ANN (pre-loaded from file by read_data)
    node_outputs[0] = summed_inputs[0] = input_symm_vec.t(); // Special case for input layer

    /* Iterate through all hidden layers. The reason we save inputs/outputs
       is because we need them during differentiation later, backprop */
    for (i = 0; i < N; ++i){
        summed_inputs[i+1] = node_outputs[i]*list_of_weights[i] + list_of_biases[i];
        node_outputs[i+1]  = act_func_sigmoid(summed_inputs[i+1]);
    }
    // Special case for output layer
    summed_inputs[N+1] = node_outputs[N] * list_of_weights[N] + list_of_biases[N];
    node_outputs[N+1]  = summed_inputs[N+1]; // No act.func.

    // Return the energy value, not 1x1 matrix:
    return as_scalar(node_outputs[N+1]);
}

mat PairNNP::backprop(){
  /* This is not ordinary backpropagation!!
     This is efficient differentiation of ANN output with respect
     to its inputs, by _use of_ backprop. We send back the
     derivative of the output node's act.func. (see thesis for deriv.) */
  mat diff = ones(1,1);

  // Loop backwards through hidden layers
  for (i = N; i > 0; --i){
      diff = diff*list_of_weights_T[i]%ddx_act_func_sigmoid(summed_inputs[i]);
  }
  // We also differentiate to the input layer: (not part of std. backprop.)
  return diff*list_of_weights_T[0];
}

mat PairNNP::act_func_sigmoid(mat node_inputs){
  return 1.0/(1 + exp(-node_inputs));
}

mat PairNNP::ddx_act_func_sigmoid(mat node_inputs){
  mat tmp = act_func_sigmoid(node_inputs);
  return tmp % (1 - tmp);
}

mat PairNNP::act_func_tanh(mat node_inputs){
  return tanh(node_inputs);
}
mat PairNNP::ddx_act_func_tanh(mat node_inputs){
  mat tmp = act_func_tanh(node_inputs);
  return 1-tmp%tmp;
}

mat PairNNP::cut_func_cos(mat R, double Rc){
  // Only receives values below cutoff
  return 0.5*(cos(M_PI*R/Rc) + 1);
}

double PairNNP::cut_func_cos_single(double R, double Rc){
  if (R < Rc){
    return 0.5*(cos(M_PI*R/Rc) + 1);
  }
  else{
    return 0;
  }
}

mat PairNNP::ddR_cut_func_cos(mat R, double Rc){
  mat tmp = -(0.5*M_PI/Rc) * sin(M_PI*R/Rc);
  return tmp;
}

double PairNNP::ddR_cut_func_cos_single(double R, double Rc){
  if (R < Rc){
    double Rcinv = 1.0/Rc;
    return -(0.5*M_PI*Rcinv) * sin(M_PI*R*Rcinv);
  }
  else return 0;
}

mat PairNNP::cut_func_tanh(mat R, double Rc){
  mat tmp = tanh(1-R/Rc);
  return tmp%tmp%tmp; // Armadillo has no elementwise power func.
}

double PairNNP::cut_func_tanh_single(double R, double Rc){
  if (R < Rc){
    return pow(tanh(1-R/Rc), 3);
  }
  else{
    return 0;
  }
}

mat PairNNP::ddR_cut_func_tanh(mat R, double Rc){
  mat tmp  = tanh(R/Rc - 1);
  mat tmp1 = tmp%tmp;   // Second power
  tmp      = tmp1%tmp1; // Fourth power, reuse tmp matrix
  return (3/Rc)*(tmp - tmp1);
}

double PairNNP::ddR_cut_func_tanh_single(double R, double Rc){
  if (R < Rc){
    double tmp = pow(tanh(R/Rc - 1),2);
    return (3/Rc)*(pow(tmp,2) - tmp);
  }
  else return 0;
}

double PairNNP::G2(double rij, double eta, double Rc, double Rs){
  return exp(-eta*pow((rij - Rs),2)) * cut_func_cos_single(rij, Rc);
}

void PairNNP::ddR_G2(mat Rij, double eta, double Rc, double Rs, mat &ddR_G2_vec){
  mat tmp = Rij - Rs;
  ddR_G2_vec = exp(-eta*tmp%tmp)%(-2*eta*tmp % cut_func_cos(Rij, Rc) + ddR_cut_func_cos(Rij, Rc));
}

double PairNNP::G5(double rij, double rik, double cos_theta, double eta,
                   double Rc, double zeta, double lambda){
  return pow(2,1-zeta)*pow(1+lambda*cos_theta,zeta)
         * exp(-eta*(rij*rij + rik*rik))
         * cut_func_cos_single(rij, Rc) * cut_func_cos_single(rik, Rc);
}

void PairNNP::ddR_G5(double xij, double yij, double zij, double xik, double yik, double zik,
                     double Rij, double Rik, double cos_theta, double eta, double Rc,
                     double zeta, double lambda, double *dGij, double *dGik){
  double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp_ij,tmp_ik,tmp_ij_ik;

  tmp1 = pow(2,1-zeta) * pow(1+lambda*cos_theta,zeta-1);
  tmp2 = tmp1*(1+lambda*cos_theta);
  tmp3 = exp(-eta*(Rij*Rij+Rik*Rik));
  tmp4 = cut_func_cos_single(Rij,Rc);
  tmp5 = cut_func_cos_single(Rik,Rc);
  tmp6 = lambda*zeta*tmp1*tmp3*tmp4*tmp5;
  tmp7 = tmp2*2*eta*tmp3*tmp4*tmp5;

  tmp_ij    = (cos_theta*tmp6/(Rij*Rij))+tmp7-tmp2*tmp3*ddR_cut_func_cos_single(Rij,Rc)*tmp5/Rij;
  tmp_ik    = (cos_theta*tmp6/(Rik*Rik))+tmp7-tmp2*tmp3*ddR_cut_func_cos_single(Rik,Rc)*tmp4/Rik;
  tmp_ij_ik = tmp6/(Rij*Rik);

  dGij[0] = -xij*tmp_ij + xik*tmp_ij_ik;
  dGij[1] = -yij*tmp_ij + yik*tmp_ij_ik;
  dGij[2] = -zij*tmp_ij + zik*tmp_ij_ik;

  dGik[0] = -xik*tmp_ik + xij*tmp_ij_ik;
  dGik[1] = -yik*tmp_ik + yij*tmp_ij_ik;
  dGik[2] = -zik*tmp_ik + zij*tmp_ij_ik;
}

void PairNNP::compute(int eflag, int vflag){
  if (eflag || vflag){
    ev_setup(eflag,vflag);
  }
  else{
    evflag = vflag_fdotr = 0;
  }
  // "Some" declarations...
  int i,ii,j,jj,a,m,n,p,sf,jnum;
  int nk,kk,k,nmbr_angles,what_symm_func;
  int *jlist;
  tagint itag,jtag,ktag;
  double fxtmp,fytmp,fztmp,fpair;
  double xtmp,ytmp,ztmp;
  double eta,Rc,Rs,zeta,lambda;
  double delxj,delyj,delzj,rsq1,rij;
  double delxk,delyk,delzk,rsq2,rik,cos_theta;
  double fj3[3],fk3[3],dGj[3],dGk[3];
  vec input_symm_vec;
  mat Rij,dRij;
  mat Rik,dRik,cos_t_vec;
  mat ddG_E,ddR_G2_vec;
  vector<mat> Rik_list;
  vector<mat> dRik_list;
  vector<mat> cos_t_list;
  vector<int> tagsj;
  vector<vector<int>> tagsk;

  double **x       = atom->x;
  double **f       = atom->f;
  tagint *tag      = atom->tag;
  int nlocal       = atom->nlocal;
  int newton_pair  = force->newton_pair;

  int inum         = list->inum;
  int *ilist       = list->ilist;
  int *numneigh    = list->numneigh;
  int **firstneigh = list->firstneigh;

  eng_vdwl = 0.0; // LAMMPS counter for total potential energy

  // Loop over all atoms
  for (ii = 0; ii < inum; ++ii){
    i = ilist[ii];
    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum  = numneigh[i];

    /* Create matrix to hold all maximum nmbr of neigh. coordinates and dx,dy,dz.
     <mat>.set_size: change size without keeping elem.s (fast!) */
    Rij.set_size(1, jnum);
    dRij.set_size(3, jnum);
    tagsj.clear();

    // Make lists of matrices ready dep. on nmbr of neigh.s
    Rik_list.resize(jnum-1);
    dRik_list.resize(jnum-1);
    cos_t_list.resize(jnum-1);
    tagsk.clear();

    // Initialize ANN input and reset counter for neighbours
    input_symm_vec = zeros(nmbr_of_ANN_inputs);
    n = 0;
    for (jj = 0; jj < jnum; ++jj){ // Loop over all neighbours
      j = jlist[jj];
      j &= NEIGHMASK; // Removes the bits that encode if j is "special" (fancy lammps encoding)
      jtag = tag[j];

      // Compute pair-wise interatomic distances
      delxj = x[j][0] - xtmp;
      delyj = x[j][1] - ytmp;
      delzj = x[j][2] - ztmp;
      rsq1  = delxj*delxj + delyj*delyj + delzj*delzj;

      if (rsq1 >= cutoff*cutoff){ // Enforce cutoff withour computing sqrt
        continue;
      }

      rij        = sqrt(rsq1);
      Rij (0, n) = rij;
      dRij(0, n) = delxj;
      dRij(1, n) = delyj;
      dRij(2, n) = delzj;
      tagsj.push_back(j); // Tags are global atomic identifiers (in case of parallelization)
      tagsk.push_back(vector<int>());

      // Compute symmetry functions that only require pair-interactions (G2)
      for (sf = 0; sf < nmbr_of_ANN_inputs; ++sf){
          what_symm_func = list_sf_params[sf].size();
          if (what_symm_func == 3 ){ // 3 means G2, 4 means G5 (identified by number of parameters)
              eta = list_sf_params[sf][0];
              Rc  = list_sf_params[sf][1];
              Rs  = list_sf_params[sf][2];
              input_symm_vec(sf) += G2(rij, eta, Rc, Rs);
        }
      }

      // Prepare to fill matrices with all atoms k (per i-j)
      Rik.set_size(1, jnum);
      dRik.set_size(3, jnum);
      cos_t_vec.set_size(1, jnum);
      nk = 0; // Counter for triplets that are found to be inside cutoff
      for (kk = jj+1; kk < jnum; ++kk){ // We loop over triplets where k > j
        k = jlist[kk];
        k &= NEIGHMASK;
        ktag = tag[k];

        delxk = x[k][0] - xtmp; // Interatomic dist. i-k
        delyk = x[k][1] - ytmp;
        delzk = x[k][2] - ztmp;
        rsq2  = delxk*delxk + delyk*delyk + delzk*delzk;

        if (rsq2 >= cutoff*cutoff){
          continue;
        }
        rik         = sqrt(rsq2);
        cos_theta   = (delxj*delxk+delyj*delyk+delzj*delzk)/(rij*rik);
        dRik(0, nk) = delxk;
        dRik(1, nk) = delyk;
        dRik(2, nk) = delzk;
        Rik(0,nk)   = rik;
        cos_t_vec(0, nk) = cos_theta;
        tagsk[n].push_back(k);

        // Compute symmetry functions that require pair and angle-interactions (G5)
        for (sf = 0; sf < nmbr_of_ANN_inputs; ++sf){
          what_symm_func = list_sf_params[sf].size();
          if (what_symm_func == 4){
            eta    = list_sf_params[sf][0];
            Rc     = list_sf_params[sf][1];
            zeta   = list_sf_params[sf][2];
            lambda = list_sf_params[sf][3];
            input_symm_vec(sf) += G5(rij,rik,cos_theta,eta,Rc,zeta,lambda);
          }
        }
        ++nk;
      }
      if (nk == 0){ // All triplets are used in computation
        ++n;        // Move on to next neighbour (i.e. j)
        continue;
      }
      // (Potentially) shrink size to fit nk number of triplets found
      Rik = Rik.cols(0,nk-1);
      dRik = dRik.cols(0,nk-1);
      cos_t_vec = cos_t_vec.cols(0,nk-1);
      // Store these for later use (force calculation)
      Rik_list[n]   = Rik;
      dRik_list[n]  = dRik;
      cos_t_list[n] = cos_t_vec;
      ++n;
    }
    Rij  = Rij.cols(0,n-1);
    dRij = dRij.cols(0,n-1);

    // Find pot. eng. for current atom with trained ANN:
    pot_eng = neural_network(input_symm_vec);

    // Add pot. eng. of current atom to the total:
    eng_vdwl += pot_eng;

    // Find how the energy changes with the symmetry vector using backprop:
    ddG_E = backprop();

    // For every symmetry function, compute derivatives:
    for (sf = 0; sf < nmbr_of_ANN_inputs; ++sf){
      what_symm_func = list_sf_params[sf].size();
      if (what_symm_func == 3 ){
        ddR_G2_vec = zeros(1,n); // Will contain d/dR(G2) for ALL neigh.s.
        eta = list_sf_params[sf][0];
        Rc  = list_sf_params[sf][1];
        Rs  = list_sf_params[sf][2];
        ddR_G2(Rij, eta, Rc, Rs, ddR_G2_vec);

        // Compute all G2 pair forces at once:
        ddR_G2_vec = -1.0*ddG_E(0,sf)*ddR_G2_vec/Rij;

        // Apply Newton's third law to all (neighbour) pairs, p:
        for (p = 0; p < n; ++p){
          fpair    = ddR_G2_vec(0,p);
          f[i][0] -= fpair*dRij(0,p); // Sum up on atom i
          f[i][1] -= fpair*dRij(1,p);
          f[i][2] -= fpair*dRij(2,p);

          // Add to specific neighbour only:
          f[tagsj[p]][0] += fpair*dRij(0,p);
          f[tagsj[p]][1] += fpair*dRij(1,p);
          f[tagsj[p]][2] += fpair*dRij(2,p);
        }
      }
      // Diff. G5 with respect to all valid triplets i,j,k:
      if (what_symm_func == 4){
        for (p = 0; p < n-1; ++p){ // Loop over all pairs, p
          nmbr_angles = size(Rik_list[p])(1);
          for (a = 0; a < nmbr_angles; ++a){ // Loop over all angles, a
            eta    = list_sf_params[sf][0];
            Rc     = list_sf_params[sf][1];
            zeta   = list_sf_params[sf][2];
            lambda = list_sf_params[sf][3];
            ddR_G5(dRij(0,p),dRij(1,p),dRij(2,p),dRik_list[p](0,a),dRik_list[p](1,a),
                   dRik_list[p](2,a),Rij(0,p),Rik_list[p](0,a),cos_t_list[p](0,a),
                   eta,Rc,zeta,lambda,dGj, dGk);

            // Compute all G5 forces from chain rule given in thesis:
            fj3[0] = -ddG_E(0,sf) * dGj[0]; // I-J, x-direction
            fj3[1] = -ddG_E(0,sf) * dGj[1]; // I-J, y-direction
            fj3[2] = -ddG_E(0,sf) * dGj[2]; // I-J, z-direction
            fk3[0] = -ddG_E(0,sf) * dGk[0]; // I-K, x-direction
            fk3[1] = -ddG_E(0,sf) * dGk[1]; // I-K, y-direction
            fk3[2] = -ddG_E(0,sf) * dGk[2]; // I-K, z-direction

            // Sum up contributions on atom i from j and k:
            f[i][0] -= fj3[0] + fk3[0];
            f[i][1] -= fj3[1] + fk3[1];
            f[i][2] -= fj3[2] + fk3[2];

            // More use of Newton's third law:
            f[tagsj[p]][0]    += fj3[0]; // Add to specific J
            f[tagsj[p]][1]    += fj3[1];
            f[tagsj[p]][2]    += fj3[2];
            f[tagsk[p][a]][0] += fk3[0]; // Add to specific K
            f[tagsk[p][a]][1] += fk3[1];
            f[tagsk[p][a]][2] += fk3[2];
          }
        }
      }
    }
  }
  if (vflag_fdotr){
    virial_fdotr_compute();
  }
}

/* ---------------------------------------------------------------------- */

void PairNNP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  // memory->create(neighshort,maxshort,"pair:neighshort");
  // map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNP::settings(int narg, char **arg)
{
   if (narg != 0){
      error->all(FLERR,"Illegal pair_style command");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNP::coeff(int narg, char **arg){
  if (!allocated){
    allocate();
  }
  if (narg != 4){
    error->all(FLERR,"Incorrect args for pair coefficients");
  } // Pair coeff must be [I J] = [* *]: (we dont allow multiple elements)
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // Convert arguments from input script: ("pair coeffs")
  read_file(arg[2]); // Path to graph and params.
  cutoff = force->numeric(FLERR,arg[3]);

  int n = atom->ntypes;
  int count = 0;
  for (int i = 1; i <= n; ++i){
    for (int j = i; j <= n; ++j){
        setflag[i][j] = 1; // Send "job done" to lammps for interaction "I J"
        ++count;
    }
  }
  if (count == 0){
    error->all(FLERR,"Incorrect args for pair coefficients");
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNP::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NN requires atom IDs");
  // if (force->newton_pair == 0)
  //   error->all(FLERR,"Pair style Stillinger-Weber requires newton pair on");

  // We use atom-centred, so a full neighbour list is needed:
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNP::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairNNP::read_file(char *file_path)
{
  int i,m,k,G;
  double w,b,p;  // Single weight, bias, symm.func.parameter
  string tmp,line, str_act_func, param_file_path;
  string graph_file_path = (string) file_path + "/graph.dat";

  ifstream graph_file;
  graph_file.open(graph_file_path.c_str(), ios::in);
  if (!graph_file.is_open()){
    error->all(FLERR,"Could not find / open graph-file. Check path!");
  }
  // Read all needed hyperparameters from the first line
  graph_file >> N >> M >> str_act_func >> nmbr_of_ANN_inputs >> nmbr_of_ANN_outputs;
  getline(graph_file, tmp);

  // We now know the size of the network:
  summed_inputs.resize(N+2);
  node_outputs.resize(N+2);
  list_of_weights.resize(N+1);
  list_sf_params.resize(nmbr_of_ANN_inputs);

  // Read all weights into list of matrices, line-by-line
  mat tmp_vec(1,M);
  for (line; getline(graph_file, line);){
    if (line.empty()){
        break;
    }
    stringstream stream(line);
    i = -1;
    while(stream >> w){
      tmp_vec(0,++i) = w;
    }
    // We temporary store the weights in the "transposed-weight-list":
    list_of_weights_T.push_back(tmp_vec);
  }

  // Read all biases into list of vectors, line-by-line
  for (line; getline(graph_file, line);){
    stringstream stream(line);
    i = -1;
    while (stream >> b){
        tmp_vec(0,++i) = b; // Reuse tmp_vec for biases (same length)
    }
    list_of_biases.push_back(tmp_vec);
  }
  list_of_biases[N] = list_of_biases[N](0); // Fix for bias node
  graph_file.close(); // All parameters read


  /* Construct weight matrices from current vectors
  by filling row-by-row. */
  // cout << "M: " << M << ". N: " << N << ". ann_inp:" << nmbr_of_ANN_inputs << endl;
  list_of_weights[0]        = zeros(nmbr_of_ANN_inputs,M);
  list_of_weights[0].row(0) = list_of_weights_T[0];

  for (k = 1; k < nmbr_of_ANN_inputs; ++k){ // Input to first hidden layer
    list_of_weights[0].row(k) = list_of_weights_T[k];
  }
  for (i=1; i < N; ++i){ // Weights between ALL hidden layers
    list_of_weights[i]        = zeros(M,M);
    list_of_weights[i].row(0) = list_of_weights_T[k];
    ++k;
    for (m=1; m < M; ++m){
      list_of_weights[i].row(m) = list_of_weights_T[k];
      ++k;
    }
  }
  // Weights to output layer
  list_of_weights[N] = zeros(nmbr_of_ANN_outputs,M);
  list_of_weights[N] = list_of_weights_T[k].t();

  // Transpose all weight matrices to speed up differentiation (backpropagation)
  list_of_weights_T.resize(N+1); // Will now store the actual transposed weight matrices
  for (i = 0; i < N+1; ++i){
      list_of_weights_T[i] = list_of_weights[i].t();
  }

  // Open file containing parameters of all symmetry functions
  ifstream sf_params_file;
  param_file_path = (string) file_path + "/parameters.dat";
  sf_params_file.open(param_file_path.c_str(), ios::in);
  if (!sf_params_file.is_open()){
    error->all(FLERR,"Could not find / open parameter-file. Check path!");
  }

  // Makes sure parameter file agrees with graph-file about number of symm. funcs:
  int check_symm_vec_agree;
  sf_params_file >> check_symm_vec_agree;
  getline(sf_params_file, tmp);
  if (check_symm_vec_agree != nmbr_of_ANN_inputs){
    cout << endl;
    cout << "Mismatch between graph and param.file! Check again!"         << endl;
    cout << "Graph file says:     " << nmbr_of_ANN_inputs   << " inputs." << endl;
    cout << "Parameter file says: " << check_symm_vec_agree << " inputs." << endl;
    error->all(FLERR," ");
  }

  G = 0; // Current symmetry function we store param. for
  for (line; getline(sf_params_file, line);){
    if (line.empty()){
      break;
    }
    stringstream stream(line);
    while (stream >> p){
        list_sf_params[G].push_back(p);
    }
    ++G;
  }
  sf_params_file.close();
}
