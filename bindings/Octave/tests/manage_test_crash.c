/*
 * These code is able to convert any crash code (segfault, abort...) into a standard exit code.
 */

#ifndef WIN32
// inspired from
// https://stackoverflow.com/questions/33693486/how-can-i-use-cmake-to-test-processes-that-are-expected-to-fail-with-an-exceptio

#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
  pid_t pid = fork();
  if (pid == -1) {
    // fork fails
    return 2;
  } else if (pid) {
    // Parent - wait child and interpret its result
    int status = 0;
    wait(&status);
    if (WIFSIGNALED(status))
      return 1;  // Signal-terminated
    if (WIFEXITED(status))
      return WEXITSTATUS(status);
    return -1;  // status not managed
  } else {
    // Child - execute wrapped command
    execvp(argv[1], argv + 1);
    exit(2);  // reached only if execvp fails
  }
}

#else /* WIN32 */
// inspired from
// https://docs.microsoft.com/en-us/windows/win32/procthread/creating-processes

#include <stdio.h>
#include <tchar.h>
#include <windows.h>

void _tmain(int argc, TCHAR* argv[]) {
  STARTUPINFO si;
  PROCESS_INFORMATION pi;
  DWORD dwWaitResult;

  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  ZeroMemory(&pi, sizeof(pi));

  if (argc != 2) {
    printf("Usage: %s [cmdline]\n", argv[0]);
    return;
  }

  // Start the child process.
  if (!CreateProcess(NULL,     // No module name (use command line)
                     argv[1],  // Command line
                     NULL,     // Process handle not inheritable
                     NULL,     // Thread handle not inheritable
                     FALSE,    // Set handle inheritance to FALSE
                     0,        // No creation flags
                     NULL,     // Use parent's environment block
                     NULL,     // Use parent's starting directory
                     &si,      // Pointer to STARTUPINFO structure
                     &pi)      // Pointer to PROCESS_INFORMATION structure
  ) {
    printf("CreateProcess failed (%d).\n", GetLastError());
    ExitProcess(2);
  }

  // Wait until child process exits.
  dwWaitResult = WaitForSingleObject(pi.hProcess, INFINITE);
  printf("WaitForSingleObject() returns value is 0X%.8X\n", dwWaitResult);

  switch (dwWaitResult) {
    case WAIT_ABANDONED:
      printf(
          "Mutex object was not released by the thread that owned the mutex "
          "object before the owning thread terminates...\n");
      break;
    case WAIT_OBJECT_0:
      printf("The child thread state was signaled !\n");
      break;
    case WAIT_TIMEOUT:
      printf("Time-out interval elapsed, and the child thread's state is nonsignaled.\n");
      break;
    case WAIT_FAILED:
      printf("WaitForSingleObject() failed, error %u\n", GetLastError());
      ExitProcess(0);
  }

  // Close process and thread handles.
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
}

#endif /* WIN32 */