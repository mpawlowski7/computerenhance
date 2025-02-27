#pragma once

#include <llama_llava_phi.h>
#include <tts_sherpa_onnx.h>

#include <QObject>

class ImageWorker: public QObject
{
    Q_OBJECT
public:
    ImageWorker();
    explicit ImageWorker(const QString& userPrompt);
    virtual ~ImageWorker();

 public slots:
    void initialize();
    void analyze(const QString& imagePath);

private:
    std::string m_userPrompt;
    std::unique_ptr<ml::LlavaPhiMini>   m_llava;
    std::unique_ptr<tts::TtsSherpaOnnx> m_ttsEngine;

signals:
    void responseReady(const QString& response);
    void doneProcessing();
};