/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <sstream>

namespace chitu {

namespace detail {

// Base case of the dispatch recursion: no candidate is left.
inline bool dispatchToStaticIntsInner(int val, auto &&lambda) { return false; }

template <int FIRST_POSSIBLE_VAL, int... OTHER_POSSIBLE_VALS>
bool dispatchToStaticIntsInner(int val, auto &&lambda) {
    if (val == FIRST_POSSIBLE_VAL) {
        lambda.template operator()<FIRST_POSSIBLE_VAL>();
        return true;
    }
    return dispatchToStaticIntsInner<OTHER_POSSIBLE_VALS...>(val, lambda);
}

// Base case of the dispatch recursion: no candidate is left.
inline bool dispatchToTypesInner(c10::ScalarType st, auto &&lambda) {
    return false;
}

template <typename FIRST_POSSIBLE_TYPE, typename... OTHER_POSSIBLE_TYPES>
bool dispatchToTypesInner(c10::ScalarType st, auto &&lambda) {
    if (st == c10::CppTypeToScalarType<FIRST_POSSIBLE_TYPE>::value) {
        lambda.template operator()<FIRST_POSSIBLE_TYPE>();
        return true;
    }
    return dispatchToTypesInner<OTHER_POSSIBLE_TYPES...>(st, lambda);
}

} // namespace detail

/**
 * Dispatch to a template integer.
 *
 * Given a run-time integer and a template lambda accepting this integer as a
 * compile-time constant as its first template argument, call the lambda with
 * the matching candidate.
 *
 * Suppose the candidates are 10, 20, 30 and the runtime value is 20, then the
 * lambda is called with the template argument 20. An exception is thrown if
 * the runtime value is not one of the candidates.
 */
template <int... POSSIBLE_VALS>
void dispatchToStaticInts(int val, auto &&lambda) {
    auto success =
        detail::dispatchToStaticIntsInner<POSSIBLE_VALS...>(val, lambda);
    if (!success) {
        std::ostringstream oss;
        oss << "Invalid value " << val
            << " passed to dispatchToStaticInts, should be one of: ";
        ((oss << POSSIBLE_VALS << ", "), ...);
        TORCH_CHECK(false, oss.str());
    }
}

/**
 * Dispatch to a template C++ scalar type.
 *
 * Given a run-time `c10::ScalarType` and a template lambda accepting the
 * corresponding C++ type as a compile-time constant as its first template
 * argument, call the lambda with the matching type. The C++ type follows the
 * ATen convention (e.g. `at::Half` for `ScalarType::Half`); kernel code maps
 * it to the CUDA type with `map_to_cuda_type`.
 *
 * Unlike the `AT_DISPATCH_*` macros, the candidate set is given explicitly,
 * so a kernel is only instantiated for the types it actually supports. An
 * exception is thrown if the runtime type is not one of the candidates.
 */
template <typename... POSSIBLE_TYPES>
void dispatchToTypes(c10::ScalarType st, auto &&lambda) {
    auto success = detail::dispatchToTypesInner<POSSIBLE_TYPES...>(st, lambda);
    if (!success) {
        std::ostringstream oss;
        oss << "Unsupported dtype " << c10::toString(st)
            << " passed to dispatchToTypes, should be one of: ";
        ((oss << c10::toString(c10::CppTypeToScalarType<POSSIBLE_TYPES>::value)
              << ", "),
         ...);
        TORCH_CHECK(false, oss.str());
    }
}

} // namespace chitu
