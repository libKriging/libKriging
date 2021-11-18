function refdir = find_dir()
    
    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0
    
    path = pwd();
    found = false;
    while ~found
        testpath = fullfile(path, ".git", "..", "tests", "references");
        disp(testpath)
        isFolder = false;
        if (isOctave)
           isFolder = exist(testpath, 'dir'); % compatible with Octave 4.2 
        else
            isFolder = isfolder(testpath); % requires R2017b
        end
        if (isFolder)
            refdir = testpath;
            return
        else
            parts = strsplit(path, filesep);
            parent = strjoin(parts(1:end-1), filesep);
            if isempty(parent) || strcmp(parent, path)
                error("Cannot find reference test directory");
            end
            path = parent;
        end
    end
end
