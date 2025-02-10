#pragma once

#include <QQmlApplicationEngine>
#include <QGuiApplication>

struct AppContext
{
    std::string name;
    std::string author;
};

namespace ui {

class QtMainWindow : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString m_prompt READ prompt() WRITE setPrompt NOTIFY promptChanged);
    Q_PROPERTY(QString m_response READ response() WRITE setResponse NOTIFY responseChanged);

public:
    explicit QtMainWindow(QObject *parent = nullptr);
    ~QtMainWindow() = default;
    void initialize(const AppContext& appCtx);
    int show();
    void setPrompt(const QString &prompt) { m_prompt = prompt; emit promptChanged(); }
    void setResponse(const QString &response) { m_response = response; emit responseChanged(); }

private:
    QQmlApplicationEngine m_engine;
    QGuiApplication       m_app;

    QString m_prompt;
    QString m_response;

signals:
    void promptChanged();
    void responseChanged();
};

} // namespace ui