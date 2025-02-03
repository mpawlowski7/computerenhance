#pragma once

#include <memory>
#include <string>

namespace tts
{

class TtsSherpaOnnx
{
public:
    TtsSherpaOnnx();
    ~TtsSherpaOnnx();

    void initialize() noexcept;
    void synthesize(const std::string& text) const noexcept;
private:
    struct TtsContext;
    std::unique_ptr<TtsContext> m_ctx;

};

}