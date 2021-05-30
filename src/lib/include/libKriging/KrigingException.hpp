#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_KRIGINGEXCEPTION_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_KRIGINGEXCEPTION_HPP

#include <exception>

class KrigingException : public std::exception {
 public:
  KrigingException(std::string msg) : m_msg(std::move(msg)) {}
  const char* what() const noexcept override { return m_msg.c_str(); }

 private:
  std::string m_msg;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_KRIGINGEXCEPTION_HPP
