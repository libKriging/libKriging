/used=/{for(i=1;i<=NF;i++) if($i ~ /^used=/){split($i,a,"=");s+=a[2];c++}} /FIT|UPDATE/{print} END{print "   solves=" c " iters_total=" s}
