// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                       \
  [&] {                                                                                          \
    if (COND) {                                                                                  \
      constexpr static bool CONST_NAME = true;                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static bool CONST_NAME = false;                                                  \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_VARLEN
  #define VARLEN_SWITCH(COND, CONST_NAME, ...)                                                   \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VARLEN_SWITCH BOOL_SWITCH
#endif

#define NUM_WARP_SWITCH(VALUE, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    if (VALUE <= 1) {                                                                            \
      constexpr static int CONST_NAME = 1;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 2) {                                                                     \
      constexpr static int CONST_NAME = 2;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 4) {                                                                     \
      constexpr static int CONST_NAME = 4;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 8) {                                                                     \
      constexpr static int CONST_NAME = 8;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 16) {                                                                    \
      constexpr static int CONST_NAME = 16;                                                      \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int CONST_NAME = 32;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
