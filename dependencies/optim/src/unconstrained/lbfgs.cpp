/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * L-BFGS method for quasi-Newton-based non-linear optimization
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::lbfgs_impl(
    Vec_t& init_out_vals, 
    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    // settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;
    const uint_t conv_failure_switch = settings.conv_failure_switch;

    const size_t iter_max = settings.iter_max;
    const double grad_err_tol = settings.grad_err_tol;
    const double rel_sol_change_tol = settings.rel_sol_change_tol;

    const double wolfe_cons_1 = settings.lbfgs_settings.wolfe_cons_1; // line search tuning parameters
    const double wolfe_cons_2 = settings.lbfgs_settings.wolfe_cons_2;

    const size_t par_M = std::max(static_cast<size_t>(2), settings.lbfgs_settings.par_M); // how many previous iterations to use when updating the Hessian

    const bool vals_bound = settings.vals_bound;
    
    const Vec_t lower_bounds = settings.lower_bounds;
    const Vec_t upper_bounds = settings.upper_bounds;

    const VecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data) \
    -> double 
    {
        if (vals_bound) {
            Vec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            double ret;
            
            if (grad_out) {
                Vec_t grad_obj = *grad_out;

                ret = opt_objfn(vals_inv_trans,&grad_obj,opt_data);

                // Mat_t jacob_matrix = jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);
                Vec_t jacob_vec = OPTIM_MATOPS_EXTRACT_DIAG( jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds) );

                // *grad_out = jacob_matrix * grad_obj; // no need for transpose as jacob_matrix is diagonal
                *grad_out = OPTIM_MATOPS_HADAMARD_PROD(jacob_vec, grad_obj);
            } else {
                ret = opt_objfn(vals_inv_trans, nullptr, opt_data);
            }

            return ret;
        } else {
            return opt_objfn(vals_inp, grad_out, opt_data);
        }
    };

    // initialization

    Vec_t x = init_out_vals;

    if (! OPTIM_MATOPS_IS_FINITE(x) ) {
        printf("lbfgs error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    Vec_t grad(n_vals);                         // gradient vector
    Vec_t d = OPTIM_MATOPS_ZERO_VEC(n_vals);    // direction vector
    Mat_t s_mat = OPTIM_MATOPS_ZERO_MAT(n_vals, par_M);
    Mat_t y_mat = OPTIM_MATOPS_ZERO_MAT(n_vals, par_M);

    box_objfn(x, &grad, opt_data);

    double grad_err = OPTIM_MATOPS_L2NORM(grad);

    OPTIM_LBFGS_TRACE(-1, grad_err, 0.0, x, d, grad, s_mat, y_mat);

    if (grad_err <= grad_err_tol) {
        return true;
    }

    // if ||gradient(initial values)|| > tolerance, then continue

    d = - grad; // direction

    Vec_t x_p = x, grad_p = grad;

    line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    Vec_t s = x_p - x;

    grad_err = OPTIM_MATOPS_L2NORM(grad);
    double rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY(s, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

    OPTIM_LBFGS_TRACE(0, grad_err, rel_sol_change, x_p, d, grad_p, s_mat, y_mat);

    if (grad_err <= grad_err_tol) {
        init_out_vals = x_p;
        return true;
    }

    // setup

    Vec_t y = grad_p - grad;

    s_mat.col(0) = s;
    y_mat.col(0) = y;

    grad = grad_p;

    // begin loop

    size_t iter = 0;

    while (grad_err > grad_err_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        //

        d = - lbfgs_recur(grad, s_mat, y_mat, std::min(iter, par_M));

        line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

        // if ||gradient(x_p)|| > tolerance, then continue

        s = x_p - x;
        y = grad_p - grad;

        grad_err = OPTIM_MATOPS_L2NORM(grad_p);
        rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY(s, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

        //

        x = x_p;
        grad = grad_p;

        OPTIM_MATOPS_MIDDLE_COLS(s_mat, 1, par_M-1) = OPTIM_MATOPS_EVAL(OPTIM_MATOPS_MIDDLE_COLS(s_mat, 0, par_M-2));
        OPTIM_MATOPS_MIDDLE_COLS(y_mat, 1, par_M-1) = OPTIM_MATOPS_EVAL(OPTIM_MATOPS_MIDDLE_COLS(y_mat, 0, par_M-2));

        s_mat.col(0) = s;
        y_mat.col(0) = y;

        //

        OPTIM_LBFGS_TRACE(iter, grad_err, rel_sol_change, x, d, grad, s_mat, y_mat)
    }

    //

    if (vals_bound) {
        x_p = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data, 
                    success, grad_err, grad_err_tol, iter, iter_max, 
                    conv_failure_switch, settings_inp);

    //
    
    return success;
}

optimlib_inline
bool
optim::lbfgs(Vec_t& init_out_vals, 
             std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
             void* opt_data)
{
    return internal::lbfgs_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::lbfgs(Vec_t& init_out_vals, 
             std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
             void* opt_data, 
             algo_settings_t& settings)
{
    return internal::lbfgs_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
