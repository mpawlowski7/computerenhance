#include "llama_llava_phi.h"

#include "llama-cpp.h"
#include "llava.h"
#include "clip.h"

#include <fmt/format.h>

struct llava_clip_deleter {
    void operator()(clip_ctx* clip) { clip_free(clip); }
};

using LLamaContextPtr = llama_context_ptr;
using LLamaModelPtr = llama_model_ptr;
using LlamaClipPtr = std::unique_ptr<clip_ctx, llava_clip_deleter>;
using LlamaParamsPtr = std::shared_ptr<llama_context_params>;

struct LlavaContext
{
    LLamaContextPtr llama;
    LLamaModelPtr   model;
    LlamaClipPtr    clip;
    LlamaParamsPtr  params;
    // std::vector<llama_token> tokensSysPrompt;
    // std::vector<llama_token> tokensUserPrompt;
};

LlavaPhiMini::LlavaPhiMini() {}

LlavaPhiMini::~LlavaPhiMini() {}

void LlavaPhiMini::initialize(const std::string& modelPath, const std::string& clipPath, int numGpuLayers)
{
    m_ctx = std::make_unique<LlavaContext>();

    initLlamaModel(modelPath, numGpuLayers);

    if (m_ctx->model == nullptr) {
        printf("%s: failed to create the m_ctx->model.get()\n", __func__);
        return;
    }

    m_ctx->params = std::make_shared<llama_context_params>();
    m_ctx->params ->n_ctx = 2048; // we need a longer context size to process image embeddings

    m_ctx->llama.reset(llama_new_context_with_model(m_ctx->model.get(), *m_ctx->params));
    if (m_ctx->llama == nullptr) {
        printf("%s: failed to create the m_ctx->llama\n", __func__);
        return;
    }

    m_ctx->clip.reset(clip_model_load(clipPath.c_str(), /*verbosity=*/ 0));
}

void LlavaPhiMini::processImage(const std::string &imagePath, const std::function<void(const std::string &response)> &
                                responseCallback)
{
    int numPast = 0;
    LlamaParamsPtr& params = m_ctx->params;
    std::string response;

    const std::string systemPrompt =
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
    const std::string userPrompt =
        "Who is at the door? Describe the look of the person, add what is he wearing, add face details. "
        "Use simple words.\nASSISTANT:";

    // bool parseSpecial, addSpecial = true;
    // responseCallback(response);

    //load image
    ImageEmbed imageEmbed = loadEmbedImage(imagePath, 4);

    // tokenize prompt
    evaluateString(systemPrompt, params->n_batch, &numPast, true);
    llava_eval_image_embed(m_ctx->llama.get(), imageEmbed, params->n_batch, &numPast);
    evaluateString(userPrompt, params->n_batch, &numPast, false);

    llava_image_embed_free(imageEmbed);
}

void LlavaPhiMini::initLlamaModel(const std::string& modelPath, int numGpuLayers)
{
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numGpuLayers;

    m_ctx->model.reset(llama_load_model_from_file(modelPath.c_str(), model_params));
    if (m_ctx->model.get() == nullptr) {
        printf("%s: unable to load model\n", __func__);
    }
}

llava_image_embed* LlavaPhiMini::loadEmbedImage(const std::string& imagePath, int numCpuThreads)
{
    ImageEmbed embed = llava_image_embed_make_with_filename(m_ctx->clip.get(), numCpuThreads,
                                                                    imagePath.c_str());
    if (!embed) {
        printf("%s: failed to embed image = %s\n",__func__, imagePath.c_str());
        return nullptr;
    }
    return embed;
}

void LlavaPhiMini::evaluateString(const std::string& prompt, int n_batch, int *n_past, bool add_bos)
{
    std::vector<llama_token> embd_inp = tokenizePrompt(prompt, true);
    evaluateTokens(embd_inp, n_batch, n_past);
}

void LlavaPhiMini::evaluateId(int id, int* n_past) // ????
{
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    evaluateTokens(tokens, 1, n_past);
}

bool LlavaPhiMini::evaluateTokens(std::vector<llama_token>& tokens, int n_batch, int* n_past) const
{
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(m_ctx->llama.get(), llama_batch_get_one(&tokens[i], n_eval))) {
           printf("%s: failed to eval. token %d/%d batch size = %d, n_past = %d",
                  __func__, i ,N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

std::vector<llama_token> LlavaPhiMini::tokenizePrompt(const std::string& prompt, bool addSpecial)
{
    int n_tokens = prompt.length() + 2 * addSpecial;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(m_ctx->model.get(), prompt.data(), prompt.length(), result.data(), result.size(), addSpecial, true);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(m_ctx->model.get(), prompt.data(), prompt.length(), result.data(), result.size(), addSpecial, true);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;;
}

void LlavaPhiMini::sampleResponse()
{
    // const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    // common_sampler_accept(smpl, id, true);
    // static std::string ret;
    // if (llama_token_is_eog(m_ctx->model.get(), id)) {
    //     ret = "</s>";
    // } else {
    //     ret = common_token_to_piece(m_ctx->llama, id);
    // }
    // evaluateId(id, n_past);
}