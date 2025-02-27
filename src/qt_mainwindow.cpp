#include <qt_mainwindow.h>
#include <qt_imageworker.h>

#include <QDebug>
#include <QThread>

static std::string g_userPrompt = "Describe the person in two sentences. Start your respond with 'The person at the door'. Focus on the gender. Focus on what is he wearing. Add face details";

namespace ui {

QtMainWindow::QtMainWindow() {}
QtMainWindow::~QtMainWindow()
{
    if (!m_workerThread.isFinished())
        m_workerThread.quit();
}

QtMainWindow::QtMainWindow(QQmlApplicationEngine* engine)
    : m_engine(engine), m_prompt(""), m_response(""), m_processing(false)
{}

void QtMainWindow::initialize()
{
    setPrompt(g_userPrompt.c_str());

    m_engine->setInitialProperties({{"ctx", QVariant::fromValue(this)}});
    m_engine->loadFromModule("MainWindow", "MainWindow");

    m_worker = std::make_unique<ImageWorker>(m_prompt);
    connect(this, &QtMainWindow::startProcessing, m_worker.get(), &ImageWorker::analyze);
    connect(m_worker.get(), &ImageWorker::responseReady, this, &QtMainWindow::setResponse, Qt::DirectConnection);
    connect(m_worker.get(), &ImageWorker::doneProcessing, [this]() {
        m_processing = false;
    });

    m_worker->moveToThread(&m_workerThread);

    connect(&m_workerThread, &QThread::started, m_worker.get(), &ImageWorker::initialize);
    m_workerThread.start();
}

void QtMainWindow::loadImage(const QString& imagePath)
{
    qDebug() << QThread::currentThreadId << __func__ ;

    m_response.clear();
    emit responseChanged();

    m_processing = true;
    emit startProcessing(imagePath);
}

}