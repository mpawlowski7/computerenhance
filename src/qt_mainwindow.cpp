#include <qt_mainwindow.h>

namespace ui {

void QtMainWindow::initialize(const AppContext& appCtx)
{
    m_app.setOrganizationName(appCtx.author.c_str());
    m_app.setApplicationName(appCtx.name.c_str());

    connect(&m_engine, &QQmlApplicationEngine::objectCreationFailed, &m_app,
        []() { QCoreApplication::exit(-1); }, Qt::QueuedConnection);

    m_engine.setInitialProperties({
            {"prompt", QVariant::fromValue(&m_prompt)},
        {"response", QVariant::fromValue(&m_response)}
    });
    m_engine.loadFromModule("MainModule", "Main");
}

int QtMainWindow::show()
{
    return m_app.exec();
}

}

