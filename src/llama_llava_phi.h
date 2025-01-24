#pragma once

#include <string>
#include <functional>
#include <memory>

struct LlavaContext;

using LlavaContextPtr = std::unique_ptr<LlavaContext>;
using TokenList = std::vector<int>;
using ImageEmbed = struct llava_image_embed*;

class LlavaPhiMini
{
public:
    LlavaPhiMini();
    ~LlavaPhiMini();

    void initialize(const std::string& modelPath, const std::string& clipPath, int numGpuLayers);
    void processImage(const std::string& imagePath, const std::function<void(const std::string& response)>& responseCallback);

private:
    LlavaContextPtr m_ctx;

    void       initLlamaModel(const std::string& modelPath, int nGpuLayers);
    ImageEmbed loadEmbedImage(const std::string& imagePath, int numCpuThreads);
    void       evaluateString(const std::string& prompt, int numBatch, int* numPast, bool addSpecial);
    void       evaluateId(int id, int* numPast);
    bool       evaluateTokens(TokenList& tokens, int numBatch, int* numPast) const;
    TokenList  tokenizePrompt(const std::string& prompt, bool addSpecial);
    void       sampleResponse();
};
