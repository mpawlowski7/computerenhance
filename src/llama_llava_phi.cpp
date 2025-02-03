#include "llama_llava_phi.h"

#include "clip.h"
#include "common.h"
#include "llama-cpp.h"
#include "llava.h"
#include "sampling.h"

#include <fmt/format.h>

namespace ml {

struct LlavaClipDeleter
{
    void operator()(clip_ctx *clip) { clip_free(clip); }
};

struct CommonSamplerDelete
{
    void operator()(common_sampler *sampler) { common_sampler_free(sampler); }
};

using LLamaContext = llama_context_ptr;
using LLamaModel = llama_model_ptr;
using LlamaClip = std::unique_ptr<clip_ctx, LlavaClipDeleter>;
using CommonSampler = std::unique_ptr<common_sampler, CommonSamplerDelete>;
using CommonParams = std::unique_ptr<common_params>;
using ImageEmbed = llava_image_embed *;

struct LlavaPhiMini::LlavaContext
{
    LLamaContext llama;
    LLamaModel model;
    LlamaClip clip;
    CommonParams params;
    CommonSampler sampler;
    TokenList tokensSysPrompt;
    TokenList tokensUserPrompt;
};

LlavaPhiMini::LlavaPhiMini() {}

LlavaPhiMini::~LlavaPhiMini()
{
    llama_backend_free();
}

void LlavaPhiMini::initialize(
    const std::string &modelPath, const std::string &clipPath, int numGpuLayers) noexcept
{
    m_ctx = std::make_unique<LlavaContext>();
    m_ctx->params = std::make_unique<common_params>();

    m_ctx->params->n_gpu_layers = numGpuLayers;
    m_ctx->params->cpuparams.n_threads = 12;

    ggml_time_init();

    common_init();

    llama_backend_init();

    llama_numa_init(m_ctx->params->numa);

    initLlamaModel(modelPath, numGpuLayers);
    if (m_ctx->model == nullptr) {
        printf("%s: failed to create the m_ctx->model\n", __func__);
        return;
    }

    llama_context_params llamaParams = common_context_params_to_llama(*m_ctx->params);
    llamaParams.n_ctx = m_ctx->params->n_ctx < 2048 ? 2048 : m_ctx->params->n_ctx;
    m_ctx->llama.reset(llama_init_from_model(m_ctx->model.get(), llamaParams));
    if (m_ctx->llama == nullptr) {
        printf("%s: failed to create the m_ctx->llama\n", __func__);
        return;
    }

    m_ctx->clip.reset(clip_model_load(clipPath.c_str(), /*verbosity=*/0));
    if (m_ctx->clip == nullptr) {
        printf("%s: failed to load clip projection\n", __func__);
        return;
    }


    const std::string systemPrompt
    = "A chat between a curious human and an artificial intelligence assistant. "
      "The assistant gives helpful and polite answers to the human's "
      "questions.\nUSER:";
    const std::string userPrompt = "Describe the person at the door in 3 sentences. Add what "
                                   "is he wearing and face details.'\nASSISTANT:";

    // Cache the prompt tokens
    m_ctx->tokensSysPrompt = tokenize(systemPrompt, true);
    m_ctx->tokensUserPrompt = tokenize(userPrompt, false);
}

void LlavaPhiMini::initLlamaModel(const std::string &modelPath, int numGpuLayers) noexcept
{
    llama_model_params modelParams = common_model_params_to_llama(*m_ctx->params);
    modelParams.n_gpu_layers = numGpuLayers;

    m_ctx->model.reset(llama_model_load_from_file(modelPath.c_str(), modelParams));
    if (m_ctx->model.get() == nullptr) {
        printf("%s: unable to load model\n", __func__);
    }
}

void LlavaPhiMini::processImage(
    const std::string &imagePath,
    const std::function<void(const std::string &response)> &responseCallback) const noexcept
{
    int numPast = 0;
    CommonParams &params = m_ctx->params;

    std::string response;

    //load image
    ImageEmbed imageEmbed = llava_image_embed_make_with_filename(
        m_ctx->clip.get(), params->cpuparams.n_threads, imagePath.c_str());
    if (!imageEmbed) {
        printf("%s: failed to embed image = %s\n", __func__, imagePath.c_str());
    }

    // tokenize system prompt
    // {
    //     TokenList tokens = tokenize(systemPrompt, true);
    //     decode(tokens, params->n_batch, &numPast);
    // }

    decode(m_ctx->tokensSysPrompt, params->n_batch, &numPast);
    // embed image
    llava_eval_image_embed(m_ctx->llama.get(), imageEmbed, params->n_batch, &numPast);

    // tokenize user prompt
    // {
    //     TokenList tokens = tokenize(userPrompt, false);
    //     decode(tokens, params->n_batch, &numPast);
    // }
    decode(m_ctx->tokensUserPrompt, params->n_batch, &numPast);

    m_ctx->sampler.reset(common_sampler_init(m_ctx->model.get(), params->sampling));
    if (!m_ctx->sampler) {
        printf("%s: failed to initialize sampling subsystem\n", __func__);
        return;
    }

    generateResponse(&numPast, params->n_batch, responseCallback);

    llava_image_embed_free(imageEmbed);
}

bool LlavaPhiMini::decode(TokenList &tokens, int n_batch, int *numPast) const noexcept
{
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(m_ctx->llama.get(), llama_batch_get_one(&tokens[i], n_eval))) {
            printf(
                "%s: failed to eval. token %d/%d batch size = %d, n_past = %d",
                __func__,
                i,
                N,
                n_batch,
                *numPast);
            return false;
        }
        *numPast += n_eval;
    }
    return true;
}

TokenList LlavaPhiMini::tokenize(const std::string &prompt, bool addBeginningOfSequence) const noexcept
{
    TokenList result = common_tokenize(m_ctx->llama.get(), prompt, addBeginningOfSequence, true);
    return result;
}

void LlavaPhiMini::generateResponse(int *numPast, int numPredict, const ResponseCallback &callback) const noexcept
{
    llama_context *llamaCtx = m_ctx->llama.get();
    common_sampler *sampler = m_ctx->sampler.get();

    std::string response;
    const int maxPredict = numPredict < 0 ? 256 : numPredict;

    for (int i = 0; i < maxPredict; i++) {
        const llama_token id = common_sampler_sample(sampler, llamaCtx, -1);
        common_sampler_accept(sampler, id, true);

        const llama_vocab *vocab = llama_model_get_vocab(m_ctx->model.get());

        static std::string ret;
        if (llama_vocab_is_eog(vocab, id)) {
            ret = "</s>";
        } else {
            ret = common_token_to_piece(llamaCtx, id);
        }
        TokenList tokens = {id};
        decode(tokens, 1, numPast);

        if (strcmp(ret.c_str(), "</s>") == 0)
            break;
        if (strstr(ret.c_str(), "###"))
            break;
        response += ret;
        if (strstr(response.c_str(), "<|im_end|>"))
            break;
        if (strstr(response.c_str(), "<|im_start|>"))
            break;
        if (strstr(response.c_str(), "USER:"))
            break;
    }
    callback(response);
}

} // namespace ml