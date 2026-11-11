#ifndef LIBKRIGING_TESTS_TEST_FORK_EXIT_HPP
#define LIBKRIGING_TESTS_TEST_FORK_EXIT_HPP

// End of a fork() child in the tests (POSIX only, include after <unistd.h>).
//
// A successful child exec()s `true` instead of calling _exit(0). Under
// valgrind (Memcheck job, --trace-children=no) an exec ends the tracing of
// the child without the leak check at exit, while the errors met in the child
// before (invalid read / write, uninitialised value) are still reported.
// That exit-time leak check is meaningless in a fork() child of a
// multithreaded process: with _exit() no destructor runs (every block of the
// process is "still reachable": Catch2 registry, ...), and the blocks held by
// the parent's other threads are "definitely lost" there because those
// threads, hence the stacks that point to them, do not exist in the child
// (~840 of the job's records at 909a0d30: ConcurrentCallsTest, BlasThreadsTest).
// Without valgrind the exit status is the same (0). A failing child keeps
// _exit(code) (its leak report then comes with a failing test anyway).
inline void lk_test_fork_child_exit(int code) {
  if (code == 0) {
    execl("/bin/true", "true", static_cast<char*>(nullptr));
    execl("/usr/bin/true", "true", static_cast<char*>(nullptr));
  }
  _exit(code);  // exec failed, or a failing child
}

#endif  // LIBKRIGING_TESTS_TEST_FORK_EXIT_HPP
