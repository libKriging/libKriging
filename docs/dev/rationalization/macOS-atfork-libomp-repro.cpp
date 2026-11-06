#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstdlib>
static int saved;
static void prep(){ saved = omp_get_max_threads(); }          // like libkriging_atfork_prepare
static void parent(){ omp_set_num_threads(saved); }
static void child(){ omp_set_num_threads(1); }
int main(int argc,char**argv){
  if(argc>1) pthread_atfork(prep,parent,child);                // registered BEFORE libomp initializes
  omp_set_num_threads(4);
  #pragma omp parallel
  { }                                                           // libomp init (registers its own atfork)
  fprintf(stderr,"before fork\n");
  pid_t p=fork();
  if(p==0){ fprintf(stderr,"child omp_get_max_threads=%d\n",omp_get_max_threads()); _exit(0);}
  int st; waitpid(p,&st,0); fprintf(stderr,"parent done, omp=%d\n",omp_get_max_threads());
}
