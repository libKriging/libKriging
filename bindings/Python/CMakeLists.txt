cmake_minimum_required(VERSION 3.1)
project(pylibkriging)

add_subdirectory(pybind11)
pybind11_add_module(pylibkriging src/main.cpp)
