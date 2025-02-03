#include <llama_llava_phi.h>
#include <tts_sherpa_onnx.h>
#include <fmt/format.h>

#include <QGuiApplication>
#include <QQmlApplicationEngine>

#define MODEL_PATH      "../models/llava-phi-3-mini-int4.gguf"
#define CLIP_PATH       "../models/llava-phi-3-mini-mmproj-f16.gguf"
#define NUM_GPU_LAYERS  25

static std::unique_ptr<tts::TtsSherpaOnnx> g_ttsEngine;

void printResponse(const std::string& response)
{
    g_ttsEngine->synthesize(response);
}

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    app.setOrganizationName("Michal Pawlowski");
    app.setApplicationName(QObject::tr("DoorbellCamera"));

    g_ttsEngine = std::make_unique<tts::TtsSherpaOnnx>();
    g_ttsEngine->initialize();

    std::unique_ptr<ml::LlavaPhiMini> llava = std::make_unique<ml::LlavaPhiMini>();
    llava->initialize(MODEL_PATH, CLIP_PATH, NUM_GPU_LAYERS);
    llava->processImage("../images/img03.jpg", printResponse);

    QQmlApplicationEngine engine;
    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreationFailed, &app,
        []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

    engine.loadFromModule("MainModule", "Main");

    printf("Starting app event loop");

    return app.exec();
}
