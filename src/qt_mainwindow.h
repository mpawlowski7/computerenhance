#pragma once

#include <QQmlApplicationEngine>
#include <QThread>

class ImageWorker;

namespace ui {

class QtMainWindow : public QObject
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(QString prompt READ prompt WRITE setPrompt NOTIFY promptChanged);
    Q_PROPERTY(QString response READ response WRITE setResponse NOTIFY responseChanged);
    Q_PROPERTY(bool processing READ processing NOTIFY processingChanged);

public:
    QtMainWindow();
    explicit QtMainWindow(QQmlApplicationEngine* appEngine);
    virtual ~QtMainWindow();

    void initialize();

    QString prompt() const { return m_prompt; }
    QString response() const { return m_response; }
    void setPrompt(const QString &prompt) { m_prompt = prompt; emit promptChanged(); }
    void setResponse(const QString &response) { m_response.append(response); emit responseChanged(); }
    bool processing() const { return m_processing; }

public slots:
    void loadImage(const QString& imagePath);

signals:
    void promptChanged();
    void responseChanged();
    void processingChanged();
    void startProcessing(const QString& imagePath);

private:
    QQmlApplicationEngine* m_engine;
    QString m_prompt;
    QString m_response;

    bool m_processing;
    QThread m_workerThread;

    std::unique_ptr<ImageWorker> m_worker;
    QElapsedTimer m_elapsedTime;
};

}
