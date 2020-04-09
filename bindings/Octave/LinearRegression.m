classdef LinearRegression < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties (NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = LinearRegression(varargin)
            printf("New LinearRegression\n");
            mLibKriging("LinearRegression::new", obj, varargin{:});
        end
        
        function delete(obj, varargin)
            % destroy the mex backend
            if ~isempty(obj.ref)
                printf("Delete LinearRegression\n")
                mLibKriging("LinearRegression::delete",obj, varargin{:})
            end
        end
        
        function fit(obj, varargin)
            mLibKriging("LinearRegression::fit", obj, varargin{:});
        end
        
        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("LinearRegression::predict", obj, varargin{:});
        end
        
    end
end
