classdef LinearRegression < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties (NonCopyable, Transient) % Access = protected, Hidden, 
        ref
    end

    methods
        function obj = LinearRegression(varargin)
            printf("New LinearRegression\n");
            mLibKriging("LinearRegression::new", obj, varargin{:});
        end
        
        function delete(obj)
            % destroy the mex backend
            if ~isempty(obj.ref)
                printf("Delete LinearRegression\n")
                mLibKriging("LinearRegression::delete",obj)
            end
        end
        
        function fit(obj, y, X)
            mLibKriging("LinearRegression::fit", obj, y, X);
        end
        
        function varargout = predict(obj, X)
            [varargout{1:nargout}] = mLibKriging("LinearRegression::predict", obj, X);
        end
        
    end
end
