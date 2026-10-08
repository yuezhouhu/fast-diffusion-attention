// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
// Adapted from FlashAttention-2 (csrc/flash_attn/src/static_switch.h).

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

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_FP16
  #define FP16_SWITCH(COND, ...)                                            \
  [&] {                                                                     \
    TORCH_CHECK(!(COND), "This build does not include the FP16 kernels");   \
    using elem_type = cutlass::bfloat16_t;                                  \
    return __VA_ARGS__();                                                   \
  }()
#else
  #define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()
#endif

// Head dimensions of the kernels: a head dimension d uses the smallest kernel >= d (32, 64, 96, 128, 192, 256), as
// FlashAttention-2; HEADDIM is the rounded head dimension (params.d_rounded). FLASHATTENTION_DISABLE_HDIM<d> removes
// a kernel from the build (development builds).
#define FDA_HDIM_CASE(HEADDIM, D, ...) if ((HEADDIM) == D) { constexpr static int kHeadDim = D; return __VA_ARGS__(); }
#ifdef FLASHATTENTION_DISABLE_HDIM32
  #define FDA_HDIM32_CASE(HEADDIM, ...)
#else
  #define FDA_HDIM32_CASE(HEADDIM, ...) FDA_HDIM_CASE(HEADDIM, 32, __VA_ARGS__)
#endif
#ifdef FLASHATTENTION_DISABLE_HDIM64
  #define FDA_HDIM64_CASE(HEADDIM, ...)
#else
  #define FDA_HDIM64_CASE(HEADDIM, ...) FDA_HDIM_CASE(HEADDIM, 64, __VA_ARGS__)
#endif
#ifdef FLASHATTENTION_DISABLE_HDIM96
  #define FDA_HDIM96_CASE(HEADDIM, ...)
#else
  #define FDA_HDIM96_CASE(HEADDIM, ...) FDA_HDIM_CASE(HEADDIM, 96, __VA_ARGS__)
#endif
#ifdef FLASHATTENTION_DISABLE_HDIM128
  #define FDA_HDIM128_CASE(HEADDIM, ...)
#else
  #define FDA_HDIM128_CASE(HEADDIM, ...) FDA_HDIM_CASE(HEADDIM, 128, __VA_ARGS__)
#endif
#ifdef FLASHATTENTION_DISABLE_HDIM192
  #define FDA_HDIM192_CASE(HEADDIM, ...)
#else
  #define FDA_HDIM192_CASE(HEADDIM, ...) FDA_HDIM_CASE(HEADDIM, 192, __VA_ARGS__)
#endif
#ifdef FLASHATTENTION_DISABLE_HDIM256
  #define FDA_HDIM256_CASE(HEADDIM, ...)
#else
  #define FDA_HDIM256_CASE(HEADDIM, ...) FDA_HDIM_CASE(HEADDIM, 256, __VA_ARGS__)
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)                                                  \
  [&] {                                                                               \
    FDA_HDIM32_CASE(HEADDIM, __VA_ARGS__) \
    FDA_HDIM64_CASE(HEADDIM, __VA_ARGS__) \
    FDA_HDIM96_CASE(HEADDIM, __VA_ARGS__) \
    FDA_HDIM128_CASE(HEADDIM, __VA_ARGS__) \
    FDA_HDIM192_CASE(HEADDIM, __VA_ARGS__) \
    FDA_HDIM256_CASE(HEADDIM, __VA_ARGS__) \
    TORCH_CHECK(false, "This build does not include the kernels for head dimension ", HEADDIM); \
  }()
