#pragma once

#include <string>
#include <functional>
#include <memory>
#include <thread>

struct common_sampler;
struct LlavaContext;

using LlavaContextPtr = std::unique_ptr<LlavaContext>;
using TokenList = std::vector<int>;
using ImageEmbed = struct llava_image_embed*;
using ResponseCallback = std::function<void(const std::string& response)>;

class LlavaPhiMini
{
public:
    LlavaPhiMini();
    ~LlavaPhiMini();

    void initialize(const std::string& modelPath, const std::string& clipPath, int numGpuLayers);
    void processImage(const std::string& imagePath, const ResponseCallback& callback);

private:
    LlavaContextPtr m_ctx;

    void       initLlamaModel(const std::string& modelPath, int numGpuLayers);
    TokenList  tokenize(const std::string& prompt, bool addSpecialToken);
    bool       decode(TokenList& tokens, int numBatch, int* numPast) const;
    void       generateResponse(int* numPast, int numPredict, ResponseCallback callback);
};
