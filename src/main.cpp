#include <llama_llava_phi.h>
#include <tts_sherpa_onnx.h>
#include <qt_mainwindow.h>
#include <fmt/format.h>

#include <QGuiApplication>

#define MODEL_PATH      "../models/llava-phi-3-mini-int4.gguf"
#define CLIP_PATH       "../models/llava-phi-3-mini-mmproj-f16.gguf"
#define NUM_GPU_LAYERS  99

static std::unique_ptr<tts::TtsSherpaOnnx> g_ttsEngine;

void printResponse(const std::string& response)
{
    if (g_ttsEngine)
        g_ttsEngine->synthesize(response);
}

int main(int argc, char *argv[])
{
    // g_ttsEngine = std::make_unique<tts::TtsSherpaOnnx>();
    // g_ttsEngine->initialize();

    // std::unique_ptr<ml::LlavaPhiMini> llava = std::make_unique<ml::LlavaPhiMini>();
    // llava->initialize(MODEL_PATH, CLIP_PATH, NUM_GPU_LAYERS);
    // llava->processImage("../images/img03.jpg", printResponse);

    // printf("Processing img01");
    // llava->processImage("../images/img01.jpg", printResponse);
    AppContext ctx;
    ctx.author = "Michal Pawlowski";


    ui::QtMainWindow appWindow;
    appWindow.initialize(ctx);

    // printf("Starting app event loop");

    return appWindow.show();
}
