#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_STRING_HASH_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_STRING_HASH_HPP

// https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
constexpr size_t fnv_hash(const char* input) {
  size_t hash = sizeof(size_t) == 8 ? 0xcbf29ce484222325 : 0x811c9dc5;
  const size_t prime = sizeof(size_t) == 8 ? 0x00000100000001b3 : 0x01000193;

  while (*input) {
    hash ^= static_cast<size_t>(*input);
    hash *= prime;
    ++input;
  }

  return hash;
}

size_t fnv_hash(const std::string& input) {
  return fnv_hash(input.c_str());
}

constexpr size_t operator"" _hash(const char* s, std::size_t) {
  return fnv_hash(s);
}

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_STRING_HASH_HPP
