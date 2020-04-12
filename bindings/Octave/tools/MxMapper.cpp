#include "MxMapper.hpp"

MxMapper::MxMapper(const char* name, const int n, mxArray** p, const RequiresArg::Requirement& requirement)
    : m_name(name), m_n(n), m_p(p) {
  assert(n < maxsize);
  assert(name != nullptr);
  if (!RequiresArg::validate(requirement, n)) {
    throw MxException(LOCATION(), "mLibKriging:badArgs", name, " requires ", RequiresArg::describe(requirement));
  }
}

MxMapper::~MxMapper() {
  for (int i = 0; i < m_n; ++i) {
    if (!m_accesses[i]) {
      mexWarnMsgIdAndTxt("mLibKriging:unusedArgument", "%s argument #%d never used", m_name, i);
    }
  }
}
