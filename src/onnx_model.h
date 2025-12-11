/**
 * ONNX model wrapper for T-one streaming CTC
 *
 * Simple wrapper over onnxruntime C++ API for running inference
 * on the T-one streaming CTC model for Russian speech recognition.
 *
 * T-one is an end-to-end model that processes raw audio samples
 * directly (no mel-filterbank feature extraction needed).
 */

#ifndef T_ONE_ONNX_MODEL_H
#define T_ONE_ONNX_MODEL_H

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace t_one {

/**
 * T-one model parameters:
 * - sample_rate: 8000 Hz
 * - frame_length: 300ms (2400 samples)
 * - frame_shift: 300ms (2400 samples) - non-overlapping
 * - output: 10 frames per input chunk (30ms each)
 *
 * Model interface:
 * - Input 1: audio samples as int32 (float * 32767), shape (batch, num_samples, 1)
 * - Input 2: state tensor, float16, shape (batch, state_dim)
 * - Output 1: log_probs, float32, shape (batch, time, vocab_size)
 * - Output 2: next_states, float16, shape (batch, state_dim)
 */
class OnnxModel {
public:
    OnnxModel();
    ~OnnxModel();

    // Non-copyable
    OnnxModel(const OnnxModel&) = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;

    /**
     * Load model from memory buffer.
     *
     * @param data Pointer to ONNX model data
     * @param size Size of model data in bytes
     * @param use_gpu If true, try to use CUDA provider
     * @return true on success
     */
    bool Load(const void* data, size_t size, bool use_gpu = true);

    /**
     * Run inference on a single audio chunk.
     *
     * The model is stateful - state is maintained internally between calls.
     * Call Reset() to clear state for a new utterance.
     *
     * @param samples Audio samples normalized to [-1, 1]
     * @param num_samples Number of samples (should be frame_length_samples for optimal processing)
     * @return Log probabilities (num_output_frames * vocab_size), row-major
     */
    std::vector<float> Forward(const float* samples, int32_t num_samples);

    /**
     * Reset internal state for new utterance.
     */
    void Reset();

    /**
     * Get vocabulary size (35 for T-one: 33 chars + space + blank)
     */
    int32_t VocabSize() const { return vocab_size_; }

    /**
     * Get expected sample rate (8000 Hz for T-one)
     */
    int32_t SampleRate() const { return sample_rate_; }

    /**
     * Get frame length in samples (2400 for T-one = 300ms at 8kHz)
     */
    int32_t FrameLengthSamples() const { return frame_length_samples_; }

    /**
     * Get state dimension
     */
    int32_t StateDim() const { return state_dim_; }

    /**
     * Check if model is loaded
     */
    bool IsLoaded() const;

    /**
     * Get last error message
     */
    const std::string& GetError() const { return error_; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;
    int32_t vocab_size_ = 35;
    int32_t sample_rate_ = 8000;
    int32_t frame_length_samples_ = 2400;  // 300ms at 8kHz
    int32_t state_dim_ = 0;
};

}  // namespace t_one

#endif  // T_ONE_ONNX_MODEL_H
