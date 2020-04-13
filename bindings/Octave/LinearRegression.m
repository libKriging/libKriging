classdef LinearRegression < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = LinearRegression(varargin)
            printf("New LinearRegression\n");
            obj.ref = mLibKriging("LinearRegression::new", varargin{:});
        end
        
        function delete(obj, varargin)
            % disp(["ObjectRef = ", num2str(obj.ref)])
            % destroy the mex backend
            if ~isempty(obj.ref)
                printf("Delete LinearRegression\n")
                obj.ref = mLibKriging("LinearRegression::delete", obj.ref, varargin{:})
            end
        end
        
        function fit(obj, varargin)
            mLibKriging("LinearRegression::fit", obj.ref, varargin{:});
        end
        
        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("LinearRegression::predict", obj.ref, varargin{:});
        end
        
    end
end
