#pragma once

#include <string>
#include <functional>
#include <memory>
#include <thread>


namespace ml
{
using TokenList = std::vector<int>;
using ResponseCallback = std::function<void(const std::string& response)>;

class LlavaPhiMini
{
public:
    LlavaPhiMini();
    ~LlavaPhiMini();

    void initialize(const std::string& modelPath, const std::string& clipPath, int numGpuLayers) noexcept;
    void processImage(const std::string& imagePath, const ResponseCallback& callback) const noexcept;

private:
    struct LlavaContext;
    std::unique_ptr<LlavaContext> m_ctx;

    void       initLlamaModel(const std::string& modelPath, int numGpuLayers) noexcept;
    TokenList  tokenize(const std::string& prompt, bool addSpecialToken) const noexcept;
    bool       decode(TokenList& tokens, int numBatch, int* numPast) const noexcept;
    void       generateResponse(int* numPast, int numPredict, const ResponseCallback& callback) const noexcept;
};

} // namespace ml
