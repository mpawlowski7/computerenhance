#include "tts_sherpa_onnx.h"

#include <QAudioSink>
#include <QBuffer>

namespace tts {

class AudioStream
{
public:
    explicit AudioStream(QAudioFormat format)
    {
        m_audioSink = std::make_unique<QAudioSink>(format);
    }
    ~AudioStream() = default;

    void start() { m_device = m_audioSink->start(); }

    void writeAudioData(const float *samples, int32_t sampleCount) const noexcept
    {
        QByteArray audioData = prepareAudioData(samples, sampleCount);
        if (m_device != nullptr) {
            m_device->write(audioData);
        }
    }

    void stop() noexcept
    {
        if (m_device != nullptr) {
            m_audioSink->stop();
            m_device = nullptr;
        }
    }

private:
    static QByteArray prepareAudioData(const float *samples, int32_t sampleCount) noexcept
    {
        QByteArray byteArray;
        byteArray.resize(sampleCount * sizeof(int16_t));

        auto pcmData = reinterpret_cast<int16_t *>(byteArray.data());

        for (size_t i = 0; i < sampleCount; ++i) {
            float sample = qBound(-1.0f, samples[i], 1.0f);
            pcmData[i] = static_cast<int16_t>(sample * INT16_MAX);
        }

        return byteArray;
    }

    std::unique_ptr<QAudioSink> m_audioSink;
    QIODevice *m_device;
};

void TtsSherpaOnnx::initialize() noexcept
{
    OfflineTtsVitsModelConfig modelConfig;
    modelConfig.model = "";
    modelConfig.lexicon = "";
    modelConfig.tokens = "";
    modelConfig.data_dir = "";
    modelConfig.dict_dir = "";

    OfflineTtsConfig config;

    m_ttsContext = std::move(OfflineTts::Create(config));
}

void TtsSherpaOnnx::synthesize(const std::string &text) const noexcept
{
    m_ttsContext.Generate(text, 0, 1, [](const float *samples, int32_t n, float progress, void *arg) {
        static std::unique_ptr<AudioStream> _audioPlayer;
        if (!_audioPlayer) {
            QAudioFormat format;
            format.setSampleRate(22050);
            format.setChannelCount(1);
            format.setSampleFormat(QAudioFormat::Int16);

            std::make_unique<AudioStream>(format);

            _audioPlayer->start();
        }

        if (progress < 1.0f) {
            _audioPlayer->writeAudioData(samples, n);

            return 1;
        }

        _audioPlayer->stop();
        return 0;
    });
}

} // namespace tts
