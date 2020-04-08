//
// Created by Pascal Havé on 07/04/2020.
//

#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_UNCOPYABLE_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_UNCOPYABLE_HPP

struct Uncopyable {
  Uncopyable() = default;
  Uncopyable(const Uncopyable&) = delete;
  Uncopyable& operator=(const Uncopyable&) = delete;
  Uncopyable(Uncopyable&&) = delete;
  Uncopyable& operator=(Uncopyable&&) = delete;
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_UNCOPYABLE_HPP
