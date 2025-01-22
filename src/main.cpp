#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include <QDebug>

#include <llama_llava_phi.h>

#define MODEL_PATH "../models/"
#define CLIP_PATH "../models/"
#define NUM_GPU_LAYERS  99

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    app.setOrganizationName("Michal Pawlowski");
    app.setApplicationName(QObject::tr("DoorbellCamera"));

    LlavaPhiMini llava;
    llava.initialize(MODEL_PATH, CLIP_PATH, NUM_GPU_LAYERS);

    QQmlApplicationEngine engine;
    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreationFailed, &app,
        []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

    engine.loadFromModule("MainModule", "Main");

    qDebug() << "Hello World";
    return app.exec();
}
