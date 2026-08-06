## Benchmark STK (https://github.com/stk-kriging/stk) against the other
## packages, on the shared datasets produced by make_datasets.py -- same
## protocol as run_python.py / run_r.R: Matern 5/2 anisotropic (ARD)
## covariance, constant trend, interpolation (no nugget beyond STK's own
## numerical safeguards), hyperparameters by MLE with STK's default
## data-driven initial guess (stk_param_estim with no param0, like
## DiceKriging/RobustGaSP).
##
## Unlike run_python.py/run_r.R, no per-fit wall-clock budget is enforced
## here: Octave has no clean equivalent of R's setTimeLimit or Python's
## subprocess timeout for interrupting a running computation mid-fit.
## Failures still surface as status=error; rely on the CI step's own
## timeout-minutes as a backstop against a stuck fit.
##
## Usage: octave run_stk.m <stk_path> <data_dir> <out_csv>
##
## Output: <out_csv> with columns
##   func,d,n,rep,package,fit_time,pred_time,rmse,q2,nlpd,status

args = argv();
stk_path = args{1};
data_dir = args{2};
out_csv = args{3};

addpath(stk_path);
stk_init;

out_dir = fileparts(out_csv);
if ~isempty(out_dir) && ~exist(out_dir, "dir")
  mkdir(out_dir);
endif

function m = read_mat(p)
  m = dlmread(p, ",");
endfunction

dirs = sort(glob(fullfile(data_dir, "*", "n*", "rep*", "X_train.csv")));
rows = {"func,d,n,rep,package,fit_time,pred_time,rmse,q2,nlpd,status"};

for i = 1:numel(dirs)
  xtr = dirs{i};
  rdir = fileparts(xtr);
  ndir = fileparts(rdir);
  fdir = fileparts(ndir);
  [~, func] = fileparts(fdir);
  [~, nname] = fileparts(ndir);
  n = str2double(nname(2:end));
  [~, repname] = fileparts(rdir);
  rep = str2double(repname(4:end));

  X = read_mat(xtr);
  y = read_mat(fullfile(rdir, "y_train.csv"));
  Xt = read_mat(fullfile(fdir, "X_test.csv"));
  yt = read_mat(fullfile(fdir, "y_test.csv"));
  d = columns(X);

  status = "ok";
  fit_time = NaN; pred_time = NaN; rmse = NaN; q2 = NaN; nlpd = NaN;
  try
    t0 = time();
    model = stk_model(@stk_materncov52_aniso, d);
    model = stk_param_estim(model, X, y);
    fit_time = time() - t0;

    t0 = time();
    zp = stk_predict(model, X, y, Xt);
    pred_time = time() - t0;

    mu = zp.mean;
    sd = sqrt(max(zp.var, 0));
    rmse = sqrt(mean((yt - mu).^2));
    q2 = 1 - sum((yt - mu).^2) / sum((yt - mean(yt)).^2);
    s2 = max(sd, 1e-12).^2;
    nlpd = mean(0.5 * log(2 * pi * s2) + 0.5 * (yt - mu).^2 ./ s2);
  catch err
    status = "error";
    fprintf(stderr, "[STK %s n=%d rep=%d] %s\n", func, n, rep, err.message);
  end_try_catch

  rows{end + 1} = sprintf("%s,%d,%d,%d,STK,%s,%s,%s,%s,%s,%s", ...
    func, d, n, rep, ...
    num2str(fit_time), num2str(pred_time), ...
    num2str(rmse), num2str(q2), num2str(nlpd), status);
  disp(rows{end});
endfor

fid = fopen(out_csv, "w");
fprintf(fid, "%s\n", strjoin(rows, "\n"));
fclose(fid);
