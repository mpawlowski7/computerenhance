#ifndef LLAMA_LLAVA_PHI_H
#define LLAMA_LLAVA_PHI_H

#include <functional>

#include "llama-cpp.h"
#include "clip.h"


struct llamaData;

class LlavaPhiMini
{
public:
    LlavaPhiMini();
    ~LlavaPhiMini();

    void initialize(const std::string& modelPath, const std::string& clipPath, int numGpuLayers);
    void processImage(const std::string& imagePath, const std::function<void(const std::string& response)>& responseCallback);

private:
    llama_context* m_pLlamaCtx;
    llama_model*   m_pLlamaModel;
    clip_ctx*      m_pClipCtx;
    std::vector<llama_token> m_tokenizedPrompt;

    void                      initLlamaModel(const std::string& modelPath, int nGpuLayers);
    struct llava_image_embed* loadEmbedImage(const std::string& imagePath, int numCpuThreads);
    void                      evaluateString(const std::string& prompt, int n_batch, int* n_past, bool add_bos);
    void                      evaluateId(int id, int* n_past);
    bool                      evaluateTokens(std::vector<llama_token>& tokens, int n_batch, int* n_past);
    std::vector<llama_token>  tokenizePrompt(const std::string& prompt, bool addBos, bool addSpecial);
    void                      sampleResponse();
};

#endif //LLAMA_LLAVA_PHI_H
