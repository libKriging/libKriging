#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_NONCOPYABLE_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_NONCOPYABLE_HPP

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable(NonCopyable&&) = delete;
  NonCopyable& operator=(NonCopyable&&) = delete;
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_NONCOPYABLE_HPP
