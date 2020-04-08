classdef LinearRegression < handle
    properties % (Access = protected, Hidden, NonCopyable, Transient)
        ref
    end

    methods
        function obj = LinearRegression()
            printf("New LinearRegression\n");
            mLibKriging("LinearRegression::new", obj);
        end
        
        function delete(obj)
            printf("Delete LinearRegression\n")
            mLibKriging("LinearRegression::delete",obj)
        end
        
        function fit(obj, y, X)
            mLibKriging("LinearRegression::fit", obj, y, X);
        end
        
        function varargout = predict(obj, X)
            [varargout{1:nargout}] = mLibKriging("LinearRegression::predict", obj, X);
        end
    end
end
