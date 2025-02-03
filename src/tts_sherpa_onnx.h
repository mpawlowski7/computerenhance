#pragma once

#include "sherpa-onnx/c-api/cxx-api.h"

#include <memory>
#include <string>

namespace tts
{
using sherpa_onnx::cxx::OfflineTts;
using sherpa_onnx::cxx::OfflineTtsConfig;
using sherpa_onnx::cxx::OfflineTtsModelConfig;
using sherpa_onnx::cxx::OfflineTtsVitsModelConfig;


class TtsSherpaOnnx
{
public:
    void initialize() noexcept;
    void synthesize(const std::string& text) const noexcept;
private:
    OfflineTts m_ttsContext;

};

}