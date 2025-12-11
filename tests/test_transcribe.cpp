/**
 * T-one Backend Test - C API
 */

#include "nerd_dictation/backend_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cctype>

/* Trim whitespace from both ends of string */
static std::string trim(const std::string& s) {
    auto start = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return (start < end) ? std::string(start, end) : std::string();
}

/* Combined context for both callbacks */
struct StreamContext {
    /* Audio state */
    const int16_t* samples;
    size_t num_samples;
    size_t offset;
    size_t chunk_size;

    /* Result state - accumulate all final segments */
    std::string all_final_text;  // All final segments concatenated
    std::string last_final_text; // Last final segment only
    std::string last_partial;
};

/* Audio callback for streaming API */
static int audio_callback(int16_t* buffer, size_t max_samples, void* user_data) {
    auto* ctx = static_cast<StreamContext*>(user_data);

    if (ctx->offset >= ctx->num_samples) {
        return 0;  /* End of stream */
    }

    size_t remaining = ctx->num_samples - ctx->offset;
    size_t to_read = std::min(std::min(ctx->chunk_size, remaining), max_samples);

    memcpy(buffer, ctx->samples + ctx->offset, to_read * sizeof(int16_t));
    ctx->offset += to_read;

    return static_cast<int>(to_read);
}

/* Result callback */
static void result_callback(const struct nerd_dictation_result* result, void* user_data) {
    auto* ctx = static_cast<StreamContext*>(user_data);

    if (result->is_final) {
        ctx->last_final_text = result->text ? result->text : "";
        // Accumulate all final segments with space separator
        if (!ctx->all_final_text.empty() && !ctx->last_final_text.empty()) {
            ctx->all_final_text += " ";
        }
        ctx->all_final_text += ctx->last_final_text;
        std::cout << "[FINAL] " << ctx->last_final_text << std::endl;
    } else if (result->text) {
        ctx->last_partial = result->text;
        std::cout << "[partial] " << result->text << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <wav_file>" << std::endl;
        return 1;
    }

    const char* wav_path = argv[1];

    std::cout << "=== T-one Backend Test (C API) ===" << std::endl;
    std::cout << "Input file: " << wav_path << std::endl;

    /* Read WAV file */
    std::ifstream file(wav_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << wav_path << std::endl;
        return 1;
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();

    if (buffer.size() < 44) {
        std::cerr << "Error: Invalid WAV file (too small)" << std::endl;
        return 1;
    }

    /* Parse WAV header */
    uint16_t num_channels = *reinterpret_cast<uint16_t*>(&buffer[22]);
    uint32_t sample_rate = *reinterpret_cast<uint32_t*>(&buffer[24]);
    uint16_t bits_per_sample = *reinterpret_cast<uint16_t*>(&buffer[34]);

    std::cout << "WAV info: " << sample_rate << " Hz, " << bits_per_sample << " bit, " << num_channels << " ch" << std::endl;

    if (bits_per_sample != 16 || num_channels != 1) {
        std::cerr << "Error: Only 16-bit mono WAV supported" << std::endl;
        return 1;
    }

    /* Extract audio data */
    size_t data_start = 44;
    const int16_t* audio_data = reinterpret_cast<const int16_t*>(buffer.data() + data_start);
    size_t num_samples = (buffer.size() - data_start) / sizeof(int16_t);

    std::cout << "Loaded " << num_samples << " samples" << std::endl;

    /* T-one model uses 8kHz */
    if (sample_rate != 8000) {
        std::cerr << "Warning: T-one model expects 8kHz audio, got " << sample_rate << " Hz" << std::endl;
    }

    /* Create backend using C API */
    std::cout << "\nCreating T-one backend..." << std::endl;

    nerd_dictation_t backend = nerd_dictation_create();
    if (!backend) {
        std::cerr << "Error: Failed to create backend" << std::endl;
        return 1;
    }

    /* Get backend info */
    struct nerd_dictation_info info;
    info.struct_size = sizeof(info);

    if (nerd_dictation_get_info(backend, &info) == 0) {
        std::cout << "Backend: " << info.name << std::endl;
        std::cout << "Streaming: " << (info.streaming ? "yes" : "no") << std::endl;
        std::cout << "Sample rate: " << info.sample_rate << " Hz" << std::endl;
    }

    /* Transcribe using streaming API */
    std::cout << "\nTranscribing..." << std::endl;

    StreamContext stream_ctx;
    stream_ctx.samples = audio_data;
    stream_ctx.num_samples = num_samples;
    stream_ctx.offset = 0;
    stream_ctx.chunk_size = 800;  /* 100ms at 8kHz */

    int ret = nerd_dictation_transcribe_stream(
        backend,
        audio_callback,
        result_callback,
        &stream_ctx
    );

    if (ret < 0) {
        std::cerr << "Error: nerd_dictation_transcribe_stream returned " << ret << std::endl;
    }

    /* Check if we got a result from streaming API */
    std::string streaming_result = stream_ctx.all_final_text;
    std::cout << "Streaming API result: \"" << streaming_result << "\"" << std::endl;

    /* Reset and use manual loop for comparison */
    nerd_dictation_reset(backend);
    stream_ctx.offset = 0;
    stream_ctx.all_final_text.clear();
    stream_ctx.last_final_text.clear();
    stream_ctx.last_partial.clear();

    std::cout << "\nTranscribing (manual loop)..." << std::endl;

    std::vector<int16_t> chunk(stream_ctx.chunk_size);
    std::string last_text;

    while (stream_ctx.offset < stream_ctx.num_samples) {
        size_t remaining = stream_ctx.num_samples - stream_ctx.offset;
        size_t to_read = std::min(stream_ctx.chunk_size, remaining);

        /* Feed audio */
        ret = nerd_dictation_feed(backend, audio_data + stream_ctx.offset, to_read);
        if (ret < 0) {
            std::cerr << "Error: nerd_dictation_feed returned " << ret << std::endl;
            nerd_dictation_destroy(backend);
            return 1;
        }
        stream_ctx.offset += to_read;

        /* Get result */
        struct nerd_dictation_result result;
        ret = nerd_dictation_result(backend, &result);
        if (ret < 0) {
            std::cerr << "Error: nerd_dictation_result returned " << ret << std::endl;
            nerd_dictation_destroy(backend);
            return 1;
        }

        if (result.text && strlen(result.text) > 0) {
            std::string current_text(result.text);

            if (result.is_final) {
                // Accumulate final segments
                if (!stream_ctx.all_final_text.empty()) {
                    stream_ctx.all_final_text += " ";
                }
                stream_ctx.all_final_text += current_text;
                std::cout << "[FINAL] " << current_text << std::endl;
            } else if (current_text != last_text) {
                std::cout << "[partial] " << current_text << std::endl;
                last_text = current_text;
            }
        }
    }

    /* Finalize */
    struct nerd_dictation_result final_result;
    ret = nerd_dictation_finalize(backend, &final_result);
    if (ret == 0 && final_result.text && strlen(final_result.text) > 0) {
        std::string finalized(final_result.text);
        // Accumulate final segment
        if (!stream_ctx.all_final_text.empty()) {
            stream_ctx.all_final_text += " ";
        }
        stream_ctx.all_final_text += finalized;
        std::cout << "[FINALIZED] " << finalized << std::endl;
    }

    /* Cleanup */
    nerd_dictation_destroy(backend);

    /* Compare with reference - use streaming_result from first pass */
    std::cout << "\n=== Result ===" << std::endl;
    std::cout << "Transcription: \"" << streaming_result << "\"" << std::endl;

    const std::string reference = "сейчас к тебе приедет бригада давай";

    std::string trimmed_result = trim(streaming_result);
    std::string trimmed_reference = trim(reference);

    std::cout << "Reference:     \"" << trimmed_reference << "\"" << std::endl;

    /* Check if result contains reference (neural networks may add filler words) */
    if (trimmed_result.find(trimmed_reference) != std::string::npos) {
        std::cout << "\n[PASS] Transcription contains reference" << std::endl;
        return 0;
    } else {
        std::cerr << "\n[FAIL] Transcription does not contain reference" << std::endl;
        return 1;
    }
}
