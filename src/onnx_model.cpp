/**
 * ONNX model implementation for T-one streaming CTC
 */

#include "onnx_model.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace t_one {

class OnnxModel::Impl {
public:
    Impl() : env_(ORT_LOGGING_LEVEL_WARNING, "t_one") {}

    bool Load(const void* data, size_t size, bool use_gpu) {
        try {
            Ort::SessionOptions session_opts;
            session_opts.SetIntraOpNumThreads(2);
            session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            if (use_gpu) {
                try {
                    OrtCUDAProviderOptions cuda_opts;
                    cuda_opts.device_id = 0;
                    session_opts.AppendExecutionProvider_CUDA(cuda_opts);
                    std::cerr << "[T-one] Using CUDA execution provider" << std::endl;
                } catch (const Ort::Exception& e) {
                    std::cerr << "[T-one] CUDA not available: " << e.what()
                              << ", falling back to CPU" << std::endl;
                }
            }

            session_ = std::make_unique<Ort::Session>(
                env_,
                data,
                size,
                session_opts
            );

            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;

            size_t num_inputs = session_->GetInputCount();
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = session_->GetInputNameAllocated(i, allocator);
                input_names_.push_back(name.get());
            }

            size_t num_outputs = session_->GetOutputCount();
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name = session_->GetOutputNameAllocated(i, allocator);
                output_names_.push_back(name.get());
            }

            // Build pointer arrays
            for (const auto& name : input_names_) {
                input_names_ptr_.push_back(name.c_str());
            }
            for (const auto& name : output_names_) {
                output_names_ptr_.push_back(name.c_str());
            }

            // Read metadata
            ReadMetadata();

            // Initialize state
            InitState();

            std::cerr << "[T-one] Model loaded: vocab_size=" << vocab_size_
                      << ", state_dim=" << state_dim_
                      << ", sample_rate=" << sample_rate_
                      << ", frame_length_ms=" << frame_length_ms_ << std::endl;

            return true;
        } catch (const Ort::Exception& e) {
            error_ = std::string("ONNX error: ") + e.what();
            std::cerr << "[T-one] " << error_ << std::endl;
            return false;
        }
    }

    std::vector<float> Forward(const float* samples, int32_t num_samples) {
        if (!session_) {
            error_ = "Model not loaded";
            return {};
        }

        try {
            // Convert float samples [-1, 1] to int32 [-32767, 32767]
            // T-one model expects: (batch, num_samples, 1) int32 tensor
            std::vector<int32_t> int_samples(num_samples);
            for (int32_t i = 0; i < num_samples; ++i) {
                float f = samples[i];
                f = std::max(-1.0f, std::min(1.0f, f));  // Clamp
                int_samples[i] = static_cast<int32_t>(f * 32767);
            }

            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtDeviceAllocator, OrtMemTypeDefault
            );

            // Input tensor: (batch=1, num_samples, 1)
            std::array<int64_t, 3> input_shape = {1, num_samples, 1};
            Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
                memory_info,
                int_samples.data(),
                int_samples.size(),
                input_shape.data(),
                input_shape.size()
            );

            // Run inference with audio and state
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(input_tensor));
            inputs.push_back(std::move(state_));

            auto outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_ptr_.data(),
                inputs.data(),
                inputs.size(),
                output_names_ptr_.data(),
                output_names_ptr_.size()
            );

            // outputs[0]: log_probs (batch, time, vocab_size)
            // outputs[1]: next_states (batch, state_dim)

            // Update state for next call
            state_ = std::move(outputs[1]);

            // Extract log_probs
            auto& log_probs = outputs[0];
            auto log_probs_shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
            size_t output_size = 1;
            for (auto dim : log_probs_shape) {
                output_size *= dim;
            }

            const float* log_probs_data = log_probs.GetTensorData<float>();
            return std::vector<float>(log_probs_data, log_probs_data + output_size);

        } catch (const Ort::Exception& e) {
            error_ = std::string("Inference error: ") + e.what();
            std::cerr << "[T-one] " << error_ << std::endl;
            return {};
        }
    }

    void Reset() {
        if (session_) {
            InitState();
        }
    }

    bool IsLoaded() const { return session_ != nullptr; }
    const std::string& GetError() const { return error_; }
    int32_t VocabSize() const { return vocab_size_; }
    int32_t StateDim() const { return state_dim_; }
    int32_t SampleRate() const { return sample_rate_; }
    int32_t FrameLengthMs() const { return frame_length_ms_; }

private:
    void ReadMetadata() {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto metadata = session_->GetModelMetadata();

            // Read frame_length_ms
            try {
                auto value = metadata.LookupCustomMetadataMapAllocated("frame_length_ms", allocator);
                if (value) {
                    frame_length_ms_ = std::stoi(value.get());
                }
            } catch (...) {}

            // Read state_dim
            try {
                auto value = metadata.LookupCustomMetadataMapAllocated("state_dim", allocator);
                if (value) {
                    state_dim_ = std::stoi(value.get());
                }
            } catch (...) {}

            // Read sample_rate
            try {
                auto value = metadata.LookupCustomMetadataMapAllocated("sample_rate", allocator);
                if (value) {
                    sample_rate_ = std::stoi(value.get());
                }
            } catch (...) {}

            // Get vocab_size from output shape
            auto output_info = session_->GetOutputTypeInfo(0);
            auto shape = output_info.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() >= 3) {
                vocab_size_ = static_cast<int32_t>(shape.back());
            }

        } catch (const Ort::Exception& e) {
            std::cerr << "[T-one] Failed to read metadata: " << e.what() << std::endl;
        }
    }

    void InitState() {
        // Create zero-initialized float16 state tensor
        // Shape: (batch=1, state_dim)
        std::array<int64_t, 2> state_shape = {1, state_dim_};

        // Allocate float16 tensor (stored as uint16_t)
        state_ = Ort::Value::CreateTensor(
            allocator_,
            state_shape.data(),
            state_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        );

        // Zero-initialize
        auto* p = state_.GetTensorMutableData<uint16_t>();
        std::fill(p, p + state_dim_, 0);
    }

    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;
    std::string error_;
    int32_t vocab_size_ = 35;
    int32_t state_dim_ = 512;  // Default, will be read from metadata
    int32_t sample_rate_ = 8000;
    int32_t frame_length_ms_ = 300;
    Ort::Value state_{nullptr};
};

OnnxModel::OnnxModel() : impl_(std::make_unique<Impl>()) {}
OnnxModel::~OnnxModel() = default;

bool OnnxModel::Load(const void* data, size_t size, bool use_gpu) {
    bool result = impl_->Load(data, size, use_gpu);
    if (result) {
        vocab_size_ = impl_->VocabSize();
        state_dim_ = impl_->StateDim();
        sample_rate_ = impl_->SampleRate();
        frame_length_samples_ = impl_->FrameLengthMs() * sample_rate_ / 1000;
    }
    error_ = impl_->GetError();
    return result;
}

std::vector<float> OnnxModel::Forward(const float* samples, int32_t num_samples) {
    auto result = impl_->Forward(samples, num_samples);
    error_ = impl_->GetError();
    return result;
}

void OnnxModel::Reset() {
    impl_->Reset();
}

bool OnnxModel::IsLoaded() const {
    return impl_->IsLoaded();
}

}  // namespace t_one
