/**
 * CTC greedy search decoder implementation for T-one
 */

#include "ctc_decoder.h"
#include <algorithm>
#include <sstream>

namespace t_one {

CtcDecoder::CtcDecoder(int32_t blank_id) : blank_id_(blank_id) {}

void CtcDecoder::Decode(const float* log_probs, int32_t num_frames, int32_t vocab_size,
                        CtcDecoderResult* result) {
    if (!result) return;

    // Get previous token for deduplication
    int32_t prev_id = -1;
    if (!result->tokens.empty()) {
        if (result->num_trailing_blanks > 0) {
            prev_id = blank_id_;
        } else {
            prev_id = result->tokens.back();
        }
    }

    const float* p = log_probs;

    for (int32_t t = 0; t < num_frames; ++t, p += vocab_size) {
        // Find argmax
        int32_t best_id = static_cast<int32_t>(
            std::distance(p, std::max_element(p, p + vocab_size))
        );

        // Track trailing blanks for endpoint detection
        if (best_id == blank_id_) {
            result->num_trailing_blanks += 1;
        } else {
            result->num_trailing_blanks = 0;
        }

        // CTC rule: skip blank and duplicate tokens
        if (best_id != blank_id_ && best_id != prev_id) {
            result->tokens.push_back(best_id);
            result->timestamps.push_back(t + result->frame_offset);
        }

        prev_id = best_id;
    }

    // Update frame offset
    result->frame_offset += num_frames;
}

void CtcDecoder::Reset(CtcDecoderResult* result) {
    if (result) {
        result->tokens.clear();
        result->timestamps.clear();
        result->num_trailing_blanks = 0;
        result->frame_offset = 0;
    }
}

std::string TokensToText(const std::vector<int32_t>& tokens,
                         const std::vector<std::string>& token_table) {
    std::string text;
    for (int32_t id : tokens) {
        if (id >= 0 && id < static_cast<int32_t>(token_table.size())) {
            text += token_table[id];
        }
    }
    return text;
}

std::vector<std::string> ParseTokens(const char* tokens_data, size_t tokens_size) {
    std::vector<std::string> result;

    std::string data(tokens_data, tokens_size);
    std::istringstream iss(data);
    std::string line;

    // First pass: find max ID
    int32_t max_id = 0;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;

        // Find last space to split token and ID
        size_t last_space = line.rfind(' ');
        if (last_space != std::string::npos) {
            int32_t id = std::stoi(line.substr(last_space + 1));
            max_id = std::max(max_id, id);
        }
    }

    // Resize result
    result.resize(max_id + 1);

    // Second pass: fill tokens
    iss.clear();
    iss.str(data);

    while (std::getline(iss, line)) {
        if (line.empty()) continue;

        size_t last_space = line.rfind(' ');
        if (last_space != std::string::npos) {
            std::string token = line.substr(0, last_space);
            int32_t id = std::stoi(line.substr(last_space + 1));

            if (id >= 0 && id < static_cast<int32_t>(result.size())) {
                result[id] = token;
            }
        }
    }

    return result;
}

}  // namespace t_one
