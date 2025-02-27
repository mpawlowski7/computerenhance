#include "qt_imageworker.h"

#include <QObject>
#include <QThread>
#include <QDebug>

static std::string g_modelPath  = "../models/llava-phi-3-mini-f16.gguf";
static std::string g_clipPath   = "../models/llava-phi-3-mini-mmproj-f16.gguf";

ImageWorker::ImageWorker() {}
ImageWorker::~ImageWorker() {}

ImageWorker::ImageWorker(const QString& userPrompt)
{
    m_userPrompt = userPrompt.toStdString();
    m_userPrompt.append("'\nASSISTANT:");
}

void ImageWorker::initialize()
{
    // LLM
    m_llava = std::make_unique<ml::LlavaPhiMini>();

    // TTS
    m_ttsEngine = std::make_unique<tts::TtsSherpaOnnx>();
    m_ttsEngine->initialize();
}

void ImageWorker::analyze(const QString& imagePath)
{
    m_llava->initialize(g_modelPath, g_clipPath, -1, m_userPrompt);

    m_llava->processImage(imagePath.toStdString(), [this](const std::string& response) {
        QString qresponse = QString::fromStdString(response);
        emit responseReady(qresponse);
        m_ttsEngine->synthesize(response.c_str());
    });

    emit doneProcessing();
}
