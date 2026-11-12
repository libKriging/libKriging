param($out, $reps = 5)
$env:PATH = "$env:USERPROFILE\Miniconda3\Library\bin;$env:PATH"; $env:OPENBLAS_L2_SIZE = "2048"
Remove-Item Env:OMP_NUM_THREADS, Env:OPENBLAS_NUM_THREADS, Env:LK_NUM_THREADS_PERCENT -ErrorAction SilentlyContinue
"toolchain,threshold,n,d,optim,rep,fit_ms,process_wall_ms,process_cpu_ms,LL,theta" | Out-File -Encoding ascii $out
foreach ($r in 1..$reps) { foreach ($n in 500,2000) { foreach ($o in 'BFGS','BFGS4') { foreach ($b in @(@('build',800),@('build-omp200',200))) {
  $exe = "C:\Users\riche\libKriging\$($b[0])\bin\bench-fit-profile.exe"
  $psi = New-Object System.Diagnostics.ProcessStartInfo $exe, "$n 4 $o"; $psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true
  $sw = [Diagnostics.Stopwatch]::StartNew(); $p = [Diagnostics.Process]::Start($psi); $txt = $p.StandardOutput.ReadToEnd(); $p.WaitForExit(); $sw.Stop()
  $cpu = $p.TotalProcessorTime.TotalMilliseconds
  $fit = [regex]::Match($txt,'Total: ([\d.]+)').Groups[1].Value; $ll = [regex]::Match($txt,'LL: ([-\d.e+]+)').Groups[1].Value; $th = [regex]::Match($txt,'theta: (.*)').Groups[1].Value.Trim()
  "msvc,$($b[1]),$n,4,$o,$r,$fit,$($sw.ElapsedMilliseconds),$([math]::Round($cpu)),$ll,$th" | Out-File -Append -Encoding ascii $out
}}}}
