% Usage example
% 
% > a = value_class ();
% > a.prop1 = 1;
% > b = a;
% > b.prop1 = 2;
% > b.prop1
% ⇒ ans =  2
% > a.prop1
% ⇒ ans =  1

% to update class definition use
% > clear classes
classdef LinearRegression < handle
    properties (Access = protected, Hidden, NonCopyable, Transient)
        ref
    end

    methods
        function obj = LinearRegression()
            printf("New LinearRegression\n");
            obj.ref = mLibKriging("LinearRegression::new");        
        end
        
        function obj = delete(obj)
            printf("Delete LinearRegression\n")
            mLibKriging("LinearRegression::delete",obj.ref)
            obj.ref = 0;
        end
    end
end
