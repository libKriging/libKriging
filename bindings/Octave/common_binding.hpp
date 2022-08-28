#include "tools/MxException.hpp"
#include "tools/MxMapper.hpp"

// Used to check if an input flag option implies the related output
bool flag_output_compliance(MxMapper& input, int I, const char* msg, const MxMapper& output, int output_position);
