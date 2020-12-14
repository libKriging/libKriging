# RlibKriging vs. DiceKriging features

| DiceKriging Feature | Implementation        | Tested       |                 |
|---------------------|-----------------------|--------------|-----------------|
| OK                  | fit(..,"constant"     |              |                 |
| UK:linear           | fit(..,"linear"       |              |                 |
| UK:quad             | fit(..,"quadratic"    |              |                 |
| UK:any              |                       |              |                 |
| warping/scaling     |                       |              |                 |
|                     |                       |              |                 |
| predict             | predict()             |              |                 |
| simulate            | simulate()            |              |                 |
| update              | update()              |              |                 |
| logLik              | logLikelihood()       |              |                 |
| loo                 | leaveOneOut()         |              |                 |
|                     |                       |              |                 |
| kernel:gauss        | (..,"gauss"           |              |                 |
| kernel:matern3_2    |                       |              |                 |
| kernel:matern5_2    |                       |              |                 |
| kernel:exp          | (..,"exp"             |              |                 |
| kernel:powexp       |                       |              |                 |
|                     |                       |              |                 |
| fit:LL              | fit(..,"LL"           |              |                 |
| fit:LOO             | fit(..,"LOO"          |              |                 |
| fit:BFGS            | fit(..,"BFGS"         |              |                 |
| fit:none            | fit(..,"none"         |              |                 |
| fit:gen             |                       |              |                 |
| fit:multistart      | fit(..,"BFGS10"       |              |                 |
|                     |                       |              |                 |
|                     |                       |              |                 |
|                     |                       |              |                 |
|                     |                       |              |                 |
