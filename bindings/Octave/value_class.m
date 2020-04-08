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
classdef value_class < handle
    properties
        prop1
    end

    methods
        function obj = value_class()
            printf("New value_class\n")        
        end
        
        function delete(obj)
            printf("Delete value_class\n")        
        end    
    
        function obj = set_prop1 (obj, val)
            printf("Set property\n")        
            obj.prop1 = val;
        end
    end
end
