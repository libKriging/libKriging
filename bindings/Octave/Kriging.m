classdef Kriging < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = Kriging(varargin)
            printf("New Kriging\n");
            obj.ref = mLibKriging("Kriging::new", varargin{:});
        end
        
        function delete(obj, varargin)
            % disp(["ObjectRef = ", num2str(obj.ref)])
            % destroy the mex backend
            if ~isempty(obj.ref)
                printf("Delete Kriging\n")
                obj.ref = mLibKriging("Kriging::delete", obj.ref, varargin{:})
            end
        end
        
        function fit(obj, varargin)
            mLibKriging("Kriging::fit", obj.ref, varargin{:});
        end
        
        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::update", obj.ref, varargin{:});
        end

        function varargout = describeModel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::describeModel", obj.ref, varargin{:});
        end

        function varargout = leaveOneOut(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::leaveOneOut", obj.ref, varargin{:});
        end

        function varargout = logLikelihood(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::logLikelihood", obj.ref, varargin{:});
        end

        function varargout = logMargPost(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::logMargPost", obj.ref, varargin{:});
        end

    end
end
