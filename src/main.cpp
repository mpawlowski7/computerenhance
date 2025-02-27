#include <qt_mainwindow.h>

#include <QGuiApplication>
#include <QThread>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app,
        []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

    ui::QtMainWindow windowContext(&engine);
    windowContext.initialize();

    return app.exec();
}
