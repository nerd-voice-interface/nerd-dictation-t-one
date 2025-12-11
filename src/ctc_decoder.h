/**
 * CTC greedy search decoder for T-one
 *
 * Implements basic CTC decoding with blank removal and deduplication.
 */

#ifndef T_ONE_CTC_DECODER_H
#define T_ONE_CTC_DECODER_H

#include <vector>
#include <string>
#include <cstdint>

namespace t_one {

/**
 * Result of CTC decoding.
 */
struct CtcDecoderResult {
    std::vector<int32_t> tokens;          // Non-blank, deduplicated tokens
    std::vector<int32_t> timestamps;      // Frame indices for each token
    int32_t num_trailing_blanks = 0;      // For endpoint detection
    int32_t frame_offset = 0;             // Total frames processed
};

/**
 * CTC greedy search decoder.
 *
 * For each frame, selects the most likely token (argmax).
 * Removes blanks and consecutive duplicate tokens.
 */
class CtcDecoder {
public:
    /**
     * Create decoder with specified blank token ID.
     *
     * @param blank_id ID of blank token in vocabulary (34 for T-one)
     */
    explicit CtcDecoder(int32_t blank_id = 34);

    /**
     * Decode log probabilities to tokens.
     *
     * @param log_probs Log probabilities (num_frames * vocab_size), row-major
     * @param num_frames Number of frames
     * @param vocab_size Vocabulary size
     * @param result Decoder result (modified in place)
     */
    void Decode(const float* log_probs, int32_t num_frames, int32_t vocab_size,
                CtcDecoderResult* result);

    /**
     * Reset decoder state for new utterance.
     */
    void Reset(CtcDecoderResult* result);

private:
    int32_t blank_id_;
};

/**
 * Convert token IDs to text using token table.
 *
 * @param tokens Token IDs
 * @param token_table Token strings indexed by ID
 * @return Decoded text
 */
std::string TokensToText(const std::vector<int32_t>& tokens,
                         const std::vector<std::string>& token_table);

/**
 * Parse tokens from T-one tokens.txt format.
 *
 * Format: "token id\n" lines, e.g., "а 0\nб 1\n..."
 *
 * @param tokens_data Raw tokens.txt content
 * @param tokens_size Size of tokens data
 * @return Token table indexed by ID
 */
std::vector<std::string> ParseTokens(const char* tokens_data, size_t tokens_size);

}  // namespace t_one

#endif  // T_ONE_CTC_DECODER_H
