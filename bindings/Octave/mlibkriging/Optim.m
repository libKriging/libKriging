classdef Optim
    % Optim - Optimization settings for libKriging
    %
    % This class provides static methods to configure optimization behavior
    % for Kriging models. All methods are static and modify global settings.
    
    methods (Static)
        function val = is_reparametrized()
            val = mLibKriging("Optim::is_reparametrized");
        end
        
        function use_reparametrize(val)
            mLibKriging("Optim::use_reparametrize", val);
        end
        
        function val = get_theta_lower_factor()
            val = mLibKriging("Optim::get_theta_lower_factor");
        end
        
        function set_theta_lower_factor(val)
            mLibKriging("Optim::set_theta_lower_factor", val);
        end
        
        function val = get_theta_upper_factor()
            val = mLibKriging("Optim::get_theta_upper_factor");
        end
        
        function set_theta_upper_factor(val)
            mLibKriging("Optim::set_theta_upper_factor", val);
        end
        
        function val = variogram_bounds_heuristic_used()
            val = mLibKriging("Optim::variogram_bounds_heuristic_used");
        end
        
        function use_variogram_bounds_heuristic(val)
            mLibKriging("Optim::use_variogram_bounds_heuristic", val);
        end
        
        function val = get_log_level()
            val = mLibKriging("Optim::get_log_level");
        end
        
        function set_log_level(val)
            mLibKriging("Optim::set_log_level", int32(val));
        end
        
        function val = get_max_iteration()
            val = mLibKriging("Optim::get_max_iteration");
        end
        
        function set_max_iteration(val)
            mLibKriging("Optim::set_max_iteration", int32(val));
        end
        
        function val = get_max_restart()
            val = mLibKriging("Optim::get_max_restart");
        end
        
        function set_max_restart(val)
            mLibKriging("Optim::set_max_restart", int32(val));
        end
        
        function val = get_gradient_tolerance()
            val = mLibKriging("Optim::get_gradient_tolerance");
        end
        
        function set_gradient_tolerance(val)
            mLibKriging("Optim::set_gradient_tolerance", val);
        end
        
        function val = get_objective_rel_tolerance()
            val = mLibKriging("Optim::get_objective_rel_tolerance");
        end
        
        function set_objective_rel_tolerance(val)
            mLibKriging("Optim::set_objective_rel_tolerance", val);
        end
        
    end
end
