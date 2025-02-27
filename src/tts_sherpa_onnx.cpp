#include "tts_sherpa_onnx.h"
#include <fmt/printf.h>

#include "sherpa-onnx/c-api/cxx-api.h"

#include <QAudioSink>
#include <QBuffer>

#define printf fmt::print

namespace tts {

using sherpa_onnx::cxx::OfflineTts;
using sherpa_onnx::cxx::OfflineTtsConfig;

struct TtsSherpaOnnx::TtsContext
{
    std::optional<OfflineTts> m_offlineTts;
    ~TtsContext()
    {
        m_offlineTts.reset();
    }
};

class AudioStream
{
public:
    explicit AudioStream(QAudioFormat format)
    {
        m_audioSink = std::make_unique<QAudioSink>(format);
        QObject::connect(m_audioSink.get(), &QAudioSink::stateChanged,
            [&](QAudio::State state) {
            switch (state) {
            case QAudio::ActiveState:
                printf("Audio stream is started\n");
                m_isStreaming = true;
                break;
            case QAudio::IdleState:
                printf("Audio stream is idle\n");
                m_isStreaming = false;
                break;
            case QAudio::StoppedState:
                printf("Audio stream is stopped\n");
                if (m_audioSink != nullptr && m_audioSink->error() != QAudio::NoError) {
                    printf("Error code =  %d\n", m_audioSink->error());
                }
                m_isStreaming = false;
                break;
            default:
                break;
            }
        });
    }
    ~AudioStream() {}

    void start()
    {
        m_device = m_audioSink->start();
        if (m_device == nullptr) {
            printf("Failed to start audio stream. Exit\n");
            exit(1);
        } else {
            auto bufferSize = m_audioSink->bufferSize();
            printf("Audio stream started. Buffer size = %d\n", bufferSize);
        }
    }

    void writeAudioData(const float *samples, int32_t sampleCount) const noexcept
    {
        QByteArray audioData = prepareAudioData(samples, sampleCount);
        if (m_device != nullptr && audioData.size() > 0) {
            m_device->write(audioData);
            m_device->waitForBytesWritten(10);
            printf("Audio stream finished.\n");
        }
    }

    void stop() noexcept
    {
        if (m_device != nullptr && m_audioSink != nullptr) {
            m_audioSink->stop();
            m_device = nullptr;
        }
    }

    bool isStreaming() const noexcept
    {
        return m_isStreaming;
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

    std::atomic<bool> m_isStreaming = false;
};

TtsSherpaOnnx::TtsSherpaOnnx() = default;
TtsSherpaOnnx::~TtsSherpaOnnx()
{

}

void TtsSherpaOnnx::initialize() noexcept
{
    m_ctx = std::make_unique<TtsContext>();

    OfflineTtsConfig config;
    config.model.vits.model = "../tts/vits-piper-en_US-amy-low/en_US-amy-low.onnx";
    config.model.vits.tokens = "../tts/vits-piper-en_US-amy-low/tokens.txt";
    config.model.vits.data_dir = "../tts/vits-piper-en_US-amy-low/espeak-ng-data";
    config.model.num_threads = 8;
    config.max_num_sentences = 1;

    m_ctx->m_offlineTts = OfflineTts::Create(config);
}

void TtsSherpaOnnx::synthesize(const std::string &text) const noexcept
{
    printf("%s\n", text.c_str());

    if (!m_ctx->m_offlineTts.has_value()) {
        printf("%s : sherpa offline tts not initialized.\n", __func__);
        return;
    }

    m_ctx->m_offlineTts.value()
        .Generate(text, 0, 1, [](const float *samples, int32_t n, float progress, void *arg) {
            Q_UNUSED(arg);

            printf("progress = %f\n", progress);
            static std::unique_ptr<AudioStream> _audioPlayer;
            if (!_audioPlayer) {
                printf("creating audio stream\n");

                QAudioFormat format;
                format.setSampleRate(16000);
                format.setChannelCount(1);
                format.setSampleFormat(QAudioFormat::Int16);

                _audioPlayer = std::make_unique<AudioStream>(format);
            }
            if (progress > 0.0f && !_audioPlayer->isStreaming()) {
                _audioPlayer->start();
            }

            _audioPlayer->writeAudioData(samples, n);

            if (progress == 1.0f) {
                _audioPlayer->stop();
                return 0;
            }

            return 1;
        });
}

} // namespace tts