function refdir = find_dir()
    path = pwd();
    found = false;
    while ~found
        testpath = [ path  filesep ".git" filesep ".." filesep "tests" filesep "references" ];
        % if isfolder(testpath) % requires R2017b 
        if exist(testpath, 'dir') % compatible with Octave 4.2
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
