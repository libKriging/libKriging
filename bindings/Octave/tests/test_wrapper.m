filepath = getenv("TESTFILE");
printf("Testing file '%s'\n", filepath);
[dir,name,ext]=fileparts(filepath);
addpath(dir)
try
    eval(name);
catch exception
    printf("An exception has been caught: %s", exception.identifier);
    exit(1);
end