/**
 * T-one ASR Backend - C API Implementation
 *
 * Uses sherpa-onnx with T-one CTC model for Russian language.
 * The model is embedded into the library and loaded from memory.
 */

#include "nerd_dictation/backend_api.h"
#include <res_embed.h>
#include <sherpa-onnx/c-api/c-api.h>
#include "models/sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.h"
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>

/* Internal backend structure */
struct nerd_dictation_backend {
    const SherpaOnnxOnlineRecognizer* recognizer;
    const SherpaOnnxOnlineStream* stream;

    int sample_rate;

    /* Embedded resources (pointers to static data, no copy) */
    const char* model_data;
    size_t model_size;

    /* Result buffer (owned, reused between calls) */
    std::string result_text;
    std::string last_partial;
    std::string error_message;

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

    backend->recognizer = nullptr;
    backend->stream = nullptr;
    backend->sample_rate = 8000;  /* T-one uses 8kHz */
    backend->model_data = nullptr;
    backend->model_size = 0;

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

    /* Configure sherpa-onnx */
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    /* Feature config (T-one uses raw audio, not mel features) */
    config.feat_config.sample_rate = backend->sample_rate;
    config.feat_config.feature_dim = 0;

    /* T-one CTC model - load from memory buffer */
    config.model_config.t_one_ctc.model_buf = backend->model_data;
    config.model_config.t_one_ctc.model_buf_size = static_cast<int32_t>(backend->model_size);

    /* Tokens - load from embedded header */
    config.model_config.tokens_buf = t_one::TOKENS_DATA;
    config.model_config.tokens_buf_size = t_one::TOKENS_SIZE;

    /* General model config */
    config.model_config.num_threads = 2;
    config.model_config.provider = "cuda";
    config.model_config.debug = 0;

    /* Decoding config */
    config.decoding_method = "greedy_search";
    config.max_active_paths = 4;

    /* Endpoint detection config */
    config.enable_endpoint = 1;
    config.rule1_min_trailing_silence = 2.4f;
    config.rule2_min_trailing_silence = 1.2f;
    config.rule3_min_utterance_length = 20.0f;

    /* Create recognizer */
    backend->recognizer = SherpaOnnxCreateOnlineRecognizer(&config);
    if (!backend->recognizer) {
        backend->error_message = "Failed to create T-one recognizer";
        std::cerr << "[T-one] ERROR: " << backend->error_message << std::endl;
        delete backend;
        return nullptr;
    }

    /* Create stream */
    backend->stream = SherpaOnnxCreateOnlineStream(backend->recognizer);
    if (!backend->stream) {
        SherpaOnnxDestroyOnlineRecognizer(backend->recognizer);
        backend->error_message = "Failed to create T-one stream";
        std::cerr << "[T-one] ERROR: " << backend->error_message << std::endl;
        delete backend;
        return nullptr;
    }

    std::cerr << "[T-one] Model loaded successfully (sample rate: "
              << backend->sample_rate << " Hz)" << std::endl;

    return backend;
}

extern "C" void nerd_dictation_destroy(nerd_dictation_t backend) {
    if (!backend) return;

    if (backend->stream) {
        SherpaOnnxDestroyOnlineStream(backend->stream);
    }
    if (backend->recognizer) {
        SherpaOnnxDestroyOnlineRecognizer(backend->recognizer);
    }

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
    if (!backend || !backend->recognizer || !backend->stream) return -1;
    if (!samples || num_samples == 0) return 0;

    /* Convert int16 to float [-1, 1] */
    std::vector<float> float_samples(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float_samples[i] = samples[i] / 32768.0f;
    }

    /* Feed audio to stream */
    SherpaOnnxOnlineStreamAcceptWaveform(
        backend->stream,
        backend->sample_rate,
        float_samples.data(),
        static_cast<int32_t>(num_samples)
    );

    /* Decode ready frames */
    while (SherpaOnnxIsOnlineStreamReady(backend->recognizer, backend->stream)) {
        SherpaOnnxDecodeOnlineStream(backend->recognizer, backend->stream);
    }

    return 0;
}

extern "C" int nerd_dictation_result(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->recognizer || !backend->stream) return -2;

    /* Check for endpoint */
    int is_endpoint = SherpaOnnxOnlineStreamIsEndpoint(backend->recognizer, backend->stream);

    /* Get result from sherpa-onnx */
    const SherpaOnnxOnlineRecognizerResult* sherpa_result =
        SherpaOnnxGetOnlineStreamResult(backend->recognizer, backend->stream);

    {
        std::lock_guard<std::mutex> lock(backend->result_mutex);

        if (sherpa_result && sherpa_result->text && strlen(sherpa_result->text) > 0) {
            backend->result_text = sherpa_result->text;
        } else {
            backend->result_text.clear();
        }

        result->text = backend->result_text.c_str();
        result->is_final = is_endpoint;
        result->confidence = -1.0f;  /* Not available */

        /* If endpoint, reset stream for next utterance */
        if (is_endpoint && !backend->result_text.empty()) {
            SherpaOnnxOnlineStreamReset(backend->recognizer, backend->stream);
            backend->last_partial.clear();
        }
    }

    if (sherpa_result) {
        SherpaOnnxDestroyOnlineRecognizerResult(sherpa_result);
    }

    return 0;
}

extern "C" int nerd_dictation_finalize(nerd_dictation_t backend, struct nerd_dictation_result* result) {
    if (!backend || !result) return -1;
    if (!backend->recognizer || !backend->stream) return -2;

    /* Signal end of input */
    SherpaOnnxOnlineStreamInputFinished(backend->stream);

    /* Decode remaining frames */
    while (SherpaOnnxIsOnlineStreamReady(backend->recognizer, backend->stream)) {
        SherpaOnnxDecodeOnlineStream(backend->recognizer, backend->stream);
    }

    /* Get final result */
    const SherpaOnnxOnlineRecognizerResult* sherpa_result =
        SherpaOnnxGetOnlineStreamResult(backend->recognizer, backend->stream);

    {
        std::lock_guard<std::mutex> lock(backend->result_mutex);

        if (sherpa_result && sherpa_result->text && strlen(sherpa_result->text) > 0) {
            backend->result_text = sherpa_result->text;
            std::cerr << "[T-one] Final result: '" << backend->result_text << "'" << std::endl;
        } else {
            backend->result_text.clear();
        }

        result->text = backend->result_text.c_str();
        result->is_final = 1;
        result->confidence = -1.0f;
    }

    if (sherpa_result) {
        SherpaOnnxDestroyOnlineRecognizerResult(sherpa_result);
    }

    return 0;
}

extern "C" int nerd_dictation_reset(nerd_dictation_t backend) {
    if (!backend) return -1;
    if (!backend->recognizer || !backend->stream) return -2;

    SherpaOnnxOnlineStreamReset(backend->recognizer, backend->stream);

    {
        std::lock_guard<std::mutex> lock(backend->result_mutex);
        backend->result_text.clear();
        backend->last_partial.clear();
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
    if (!backend->recognizer || !backend->stream) return -2;

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
