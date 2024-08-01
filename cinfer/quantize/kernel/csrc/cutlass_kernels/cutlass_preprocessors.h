#pragma once
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fastertransformer {

enum class QuantType { INT8_WEIGHT_ONLY, PACKED_INT4_WEIGHT_ONLY };

int get_bits_in_quant_type(QuantType quant_type);

void preprocess_weights(int8_t *preprocessed_quantized_weight,
                        const int8_t *row_major_quantized_weight, size_t rows,
                        size_t cols, bool is_int4, int arch);

template<typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t*                    processed_quantized_weight,
                        ComputeType*               scale_ptr,
                        const WeightType*          input_weight_ptr,
                        const std::vector<size_t>& shape,
                        QuantType                  quant_type);


template<typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t*                    processed_quantized_weight,
                        int8_t*                    unprocessed_quantized_weight,
                        ComputeType*               scale_ptr,
                        const WeightType*          input_weight_ptr,
                        const std::vector<size_t>& shape,
                        QuantType                  quant_type);
} // namespace fastertransformer
