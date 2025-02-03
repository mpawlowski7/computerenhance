#include <llama_llava_phi.h>
#include <fmt/format.h>

#include <QGuiApplication>
#include <QQmlApplicationEngine>

#define MODEL_PATH "../models/llava-phi-3-mini-f16.gguf"
#define CLIP_PATH "../models/llava-phi-3-mini-mmproj-f16.gguf"
#define NUM_GPU_LAYERS 8

void printResponse(const std::string& response)
{
    printf("%s \n", response.c_str());
}

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    app.setOrganizationName("Michal Pawlowski");
    app.setApplicationName(QObject::tr("DoorbellCamera"));

    std::unique_ptr<LlavaPhiMini> llava = std::make_unique<LlavaPhiMini>();
    llava->initialize(MODEL_PATH, CLIP_PATH, NUM_GPU_LAYERS);
    llava->processImage("../images/img02.jpg", printResponse);

    // QQmlApplicationEngine engine;
    // QObject::connect(
    //     &engine, &QQmlApplicationEngine::objectCreationFailed, &app,
    //     []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);
    //
    // engine.loadFromModule("MainModule", "Main");

    printf("Starting app event loop");

    // return app.exec();
    return 0;
}
