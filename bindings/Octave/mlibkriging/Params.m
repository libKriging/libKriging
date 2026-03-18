classdef Params < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = Params(varargin)
            % fprintf("New Params with %d args\n", nargin);
            obj.ref = mLibKriging("Params::new", varargin{:});
        end
        
        function delete(obj, varargin)
            % disp(["ObjectRef = ", num2str(obj.ref)])
            % destroy the mex backend
            if ~isempty(obj.ref)
                % fprintf("Delete Params\n")
                obj.ref = mLibKriging("Params::delete", obj.ref, varargin{:})
            end
        end
        
        function display(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Params::display", obj.ref, varargin{:});
        end
    end
end
