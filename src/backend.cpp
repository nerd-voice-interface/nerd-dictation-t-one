/**
 * T-one ASR Backend - C API Implementation
 *
 * Uses direct ONNX Runtime inference with T-one CTC model for Russian language.
 * The model is embedded into the library and loaded from memory.
 */

#include "nerd_dictation/backend_api.h"
#include "onnx_model.h"
#include "ctc_decoder.h"
#include <res_embed.h>
#include "models/sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.h"
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>

/* Internal backend structure */
struct nerd_dictation_backend {
    std::unique_ptr<t_one::OnnxModel> model;
    std::unique_ptr<t_one::CtcDecoder> decoder;
    t_one::CtcDecoderResult decoder_result;

    int sample_rate;
    int frame_length_samples;

    /* Token table for decoding */
    std::vector<std::string> token_table;

    /* Audio buffer for accumulating samples until we have enough for a frame */
    std::vector<float> audio_buffer;

    /* Embedded resources (pointers to static data, no copy) */
    const char* model_data;
    size_t model_size;

    /* Result buffer (owned, reused between calls) */
    std::string result_text;
    std::string error_message;

    /* Endpoint detection parameters (in frames, 30ms each) */
    int trailing_silence_threshold;  // Number of blank frames for endpoint

    /* Thread safety for result buffer */
    std::mutex result_mutex;
};

/* ============================================================================
 * Lifecycle functions
 * ============================================================================ */

extern "C" nerd_dictation_t nerd_dictation_create(void) {
    auto* backend = new (std::nothrow) nerd_dictation_backend();
    if (!backend) {
        return nullptr;
    }

    backend->sample_rate = 8000;  /* T-one uses 8kHz */
    backend->frame_length_samples = 2400;  /* 300ms at 8kHz */
    backend->model_data = nullptr;
    backend->model_size = 0;
    backend->trailing_silence_threshold = 40;  // 40 frames * 30ms = 1.2s

    std::cerr << "[T-one] Loading embedded model from memory..." << std::endl;

    /* Get embedded model */
    backend->model_data = res::embed::get("t_one_model_onnx", &backend->model_size);

    if (!backend->model_data || backend->model_size == 0) {
        backend->error_message = "T-one: embedded model not found";
        std::cerr << "[T-one] ERROR: " << backend->error_message << std::endl;
        delete backend;
        return nullptr;
    }

    std::cerr << "[T-one] Model size: " << backend->model_size
              << " bytes, tokens size: " << t_one::TOKENS_SIZE << " bytes" << std::endl;

    /* Create and load model */
    backend->model = std::make_unique<t_one::OnnxModel>();
    if (!backend->model->Load(backend->model_data, backend->model_size, true)) {
        backend->error_message = "Failed to load T-one model: " + backend->model->GetError();
        std::cerr << "[T-one] ERROR: " << backend->error_message << std::endl;
        delete backend;
        return nullptr;
    }

    /* Update parameters from model metadata */
    backend->sample_rate = backend->model->SampleRate();
    backend->frame_length_samples = backend->model->FrameLengthSamples();

    /* Parse tokens */
    backend->token_table = t_one::ParseTokens(t_one::TOKENS_DATA, t_one::TOKENS_SIZE);

    /* Create decoder with blank_id = 34 */
    backend->decoder = std::make_unique<t_one::CtcDecoder>(34);

    std::cerr << "[T-one] Model loaded successfully (sample rate: "
              << backend->sample_rate << " Hz, frame: "
              << backend->frame_length_samples << " samples)" << std::endl;

    return backend;
}

extern "C" void nerd_dictation_destroy(nerd_dictation_t backend) {
    if (!backend) return;
    delete backend;
}

extern "C" int nerd_dictation_get_info(nerd_dictation_t backend, struct nerd_dictation_info* info) {
    if (!backend || !info) return -1;
    if (info->struct_size < sizeof(struct nerd_dictation_info)) return -2;

    info->name = "t-one";
    info->streaming = 1;
    info->sample_rate = backend->sample_rate;
    info->channels = 1;
    info->bits_per_sample = 16;

    return 0;
}

/* ============================================================================
 * Recognition functions
 * ============================================================================ */

extern "C" int nerd_dictation_feed(nerd_dictation_t backend, const int16_t* samples, size_t num_samples) {
    if (!backend || !backend->model || !backend->decoder) return -1;
    if (!samples || num_samples == 0) return 0;

    /* Convert int16 to float [-1, 1] and add to buffer */
    for (size_t i = 0; i < num_samples; ++i) {
        backend->audio_buffer.push_back(samples[i] / 32768.0f);
    }

    /* Process complete frames */
    while (backend->audio_buffer.size() >= static_cast<size_t>(backend->frame_length_samples)) {
        /* Run inference on frame */
        std::vector<float> log_probs = backend->model->Forward(
            backend->audio_buffer.data(),
            backend->frame_length_samples
        );

        if (log_probs.empty()) {
            return -2;  // Inference error
        }

        /* Decode */
        int32_t num_frames = static_cast<int32_t>(log_probs.size()) / backend->model->VocabSize();
        backend->decoder->Decode(
            log_probs.data(),
            num_frames,
            backend->model->VocabSize(),
            &backend->decoder_result
        );

        /* Remove processed samples from buffer */
        backend->audio_buffer.erase(
            backend->audio_buffer.begin(),
            backend->audio_buffer.begin() + backend->frame_length_samples
        );
    }

    return 0;
}

extern "C" int nerd_dictation_result(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->model || !backend->decoder) return -2;

    /* Check for endpoint (trailing silence) */
    int is_endpoint = backend->decoder_result.num_trailing_blanks >= backend->trailing_silence_threshold;

    {
        std::lock_guard<std::mutex> lock(backend->result_mutex);

        /* Convert tokens to text */
        backend->result_text = t_one::TokensToText(
            backend->decoder_result.tokens,
            backend->token_table
        );

        result->text = backend->result_text.c_str();
        result->is_final = is_endpoint && !backend->result_text.empty();
        result->confidence = -1.0f;  /* Not available */

        /* If endpoint with text, reset for next utterance */
        if (result->is_final) {
            backend->decoder->Reset(&backend->decoder_result);
            backend->model->Reset();
        }
    }

    return 0;
}

extern "C" int nerd_dictation_finalize(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->model || !backend->decoder) return -2;

    /* Process any remaining audio in buffer */
    if (!backend->audio_buffer.empty()) {
        /* Pad with zeros if needed */
        size_t pad_size = backend->frame_length_samples - backend->audio_buffer.size();
        if (pad_size > 0) {
            backend->audio_buffer.resize(backend->frame_length_samples, 0.0f);
        }

        /* Run final inference */
        std::vector<float> log_probs = backend->model->Forward(
            backend->audio_buffer.data(),
            backend->frame_length_samples
        );

        if (!log_probs.empty()) {
            int32_t num_frames = static_cast<int32_t>(log_probs.size()) / backend->model->VocabSize();
            backend->decoder->Decode(
                log_probs.data(),
                num_frames,
                backend->model->VocabSize(),
                &backend->decoder_result
            );
        }

        backend->audio_buffer.clear();
    }

    {
        std::lock_guard<std::mutex> lock(backend->result_mutex);

        /* Convert tokens to text */
        backend->result_text = t_one::TokensToText(
            backend->decoder_result.tokens,
            backend->token_table
        );

        std::cerr << "[T-one] Final result: '" << backend->result_text << "'" << std::endl;

        result->text = backend->result_text.c_str();
        result->is_final = 1;
        result->confidence = -1.0f;
    }

    return 0;
}

extern "C" int nerd_dictation_reset(nerd_dictation_t backend) {
    if (!backend) return -1;
    if (!backend->model || !backend->decoder) return -2;

    backend->decoder->Reset(&backend->decoder_result);
    backend->model->Reset();
    backend->audio_buffer.clear();

    {
        std::lock_guard<std::mutex> lock(backend->result_mutex);
        backend->result_text.clear();
    }

    return 0;
}

/* ============================================================================
 * Streaming API
 * ============================================================================ */

extern "C" int nerd_dictation_transcribe_stream(
    nerd_dictation_t backend,
    nerd_dictation_audio_callback audio_cb,
    nerd_dictation_result_callback result_cb,
    void* user_data
) {
    if (!backend || !audio_cb) return -1;
    if (!backend->model || !backend->decoder) return -2;

    std::cerr << "[T-one] Starting streaming recognition" << std::endl;

    /* Audio buffer for callback */
    std::vector<int16_t> audio_chunk(backend->sample_rate / 10);  /* 100ms chunks */
    size_t chunk_count = 0;
    std::string last_text;

    while (true) {
        /* Get audio from callback */
        int samples_read = audio_cb(audio_chunk.data(), audio_chunk.size(), user_data);

        if (samples_read < 0) {
            /* Error */
            backend->error_message = "Audio callback returned error";
            return -3;
        }

        if (samples_read == 0) {
            /* End of stream */
            break;
        }

        chunk_count++;

        /* Feed audio */
        int ret = nerd_dictation_feed(backend, audio_chunk.data(), static_cast<size_t>(samples_read));
        if (ret < 0) {
            return ret;
        }

        /* Get result */
        struct nerd_dictation_result result;
        ret = nerd_dictation_result(backend, &result);
        if (ret < 0) {
            return ret;
        }

        /* Call result callback if text changed or is final */
        if (result_cb && result.text) {
            std::string current_text(result.text);

            if (result.is_final || current_text != last_text) {
                result_cb(&result, user_data);

                if (result.is_final) {
                    last_text.clear();
                } else {
                    last_text = current_text;
                }
            }
        }
    }

    /* Finalize and get last result */
    struct nerd_dictation_result final_result;
    int ret = nerd_dictation_finalize(backend, &final_result);
    if (ret == 0 && result_cb && final_result.text && strlen(final_result.text) > 0) {
        result_cb(&final_result, user_data);
    }

    std::cerr << "[T-one] Streaming recognition finished (processed "
              << chunk_count << " chunks)" << std::endl;

    return 0;
}

/* ============================================================================
 * Utilities
 * ============================================================================ */

extern "C" const char* nerd_dictation_get_error(nerd_dictation_t backend) {
    if (!backend) return nullptr;
    return backend->error_message.empty() ? nullptr : backend->error_message.c_str();
}

extern "C" int nerd_dictation_api_version(void) {
    return NERD_DICTATION_API_VERSION;
}
