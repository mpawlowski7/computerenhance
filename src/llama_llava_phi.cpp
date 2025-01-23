#include "llama_llava_phi.h"

#include "clip.h"
#include "llava.h"

#include <fmt/format.h>
#include <fmt/printf.h>

// struct llamaData
// {
//     llama_context* m_pLlamaCtx;
//     llama_model*   m_pLlamaModel;
//     clip_ctx*      m_pClipCtx;
//
//     ~llamaData()
//     {
//         if (m_pLlamaCtx);
//     }
// }

void LlavaPhiMini::initialize(const std::string& modelPath, const std::string& clipPath, int numGpuLayers)
{
    initLlamaModel(modelPath, numGpuLayers);
    if (m_pLlamaModel == nullptr) {
        printf("%s: failed to create the m_pLlamaModel\n", __func__);
        return;
    }

    llama_context_params ctxParams = {};
    ctxParams.n_ctx = 2048; // we need a longer context size to process image embeddings

    m_pLlamaCtx = llama_new_context_with_model(m_pLlamaModel, ctxParams);
    if (m_pLlamaCtx == nullptr) {
        printf("%s: failed to create the m_pLlamaCtx\n", __func__);
        return;
    }

    m_pClipCtx = clip_model_load(clipPath.data(), /*verbosity=*/ 0);
}

void LlavaPhiMini::processImage(const std::string &imagePath, const std::function<void(const std::string &response)> &
                                responseCallback)
{
    int numPast = 0;

    const std::string systemPrompt =
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
    const std::string userPrompt =
        "Who is at the door? Describe the look of the person, add what is he wearing, add face details. "
        "Use simple words.\nASSISTANT:";

    std::string response;
    // bool parseSpecial, addSpecial = true;
    // responseCallback(response);

    //load image
    llava_image_embed *imageEmbed = loadEmbedImage(imagePath, 4);

    // tokenize prompt

    llava_eval_image_embed(m_pLlamaCtx, imageEmbed, params->n_batch, &numPast);


    llava_image_embed_free(imageEmbed);
}

void LlavaPhiMini::initLlamaModel(const std::string& modelPath, int numGpuLayers)
{
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numGpuLayers;

    m_pLlamaModel = llama_load_model_from_file(modelPath.c_str(), model_params);
    if (m_pLlamaModel == nullptr) {
        printf("%s: unable to load model\n", __func__);
        return;
    }
}

llava_image_embed* LlavaPhiMini::loadEmbedImage(const std::string& imagePath, int numCpuThreads)
{
    llava_image_embed *embed = llava_image_embed_make_with_filename(m_pClipCtx, numCpuThreads,
                                                                    imagePath.c_str()());
    if (!embed) {
        printf("%s: failed to embed image = %s\n",__func__, imagePath);
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
        if (llama_decode(m_pLlamaCtx, llama_batch_get_one(&tokens[i], n_eval))) {
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
    n_tokens = llama_tokenize(m_pLlamaModel, prompt.data(), prompt.length(), result.data(), result.size(), addSpecial, true);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(m_pLlamaModel, prompt.data(), prompt.length(), result.data(), result.size(), addSpecial, true);
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
    // if (llama_token_is_eog(m_pLlamaModel, id)) {
    //     ret = "</s>";
    // } else {
    //     ret = common_token_to_piece(m_pLlamaCtx, id);
    // }
    // evaluateId(id, n_past);
}