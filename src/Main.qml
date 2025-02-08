import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: root

    width: 1120
    height: 900
    minimumWidth: width
    minimumHeight: height
    maximumWidth: width
    maximumHeight: height
    visible: true
    flags: Qt.Window
    title: qsTr("Doorbell Camera AI - Test app")

    Material.theme: Material.Light

    Column {
        anchors.fill: parent
        spacing: 30
        anchors.topMargin: 30

        Pane {
            anchors.horizontalCenter: parent.horizontalCenter
            width: 960
            height: 540

            Material.elevation: 6
            Material.background: Material.color(Material.Grey, Material.Shade200)
            Material.roundedScale: Material.SmallScale

            Image {
                source: "file:////home/mcpw/Projects/ComputerEnhance/out/images/dummy.png"
                width: 960
                height: 540
                fillMode: Image.PreserveAspectFit
                antialiasing: true

                Button {
                    text: "Load image"
                    font.weight: Font.Medium
                    font.pixelSize: 18
                    Material.roundedScale: Material.SmallScale
                    height: 60
                }
            }
        }

        Column {
            width: 960
            height: 140
            spacing: 10

            anchors.horizontalCenter: parent.horizontalCenter

            Label {
                text: "Prompt:"
                font.weight: Font.Medium
                font.pixelSize: 18
            }

            Item {
                width: parent.width
                height: parent.height / 2

                Label {
                    text: "Describe the person at the door in three sentences. What is wearing, add face details."
                    font.pixelSize: 18

                    anchors.fill: parent
                    anchors.margins: 12
                    wrapMode: Label.Wrap
                }
            }

            Label {
                text: "Response:"
                font.weight: Font.Medium
                font.pixelSize: 18
            }

            Item {
                width: parent.width
                height: parent.height / 2

                Label {
                    id: agent_resp
                    text: "Describe the person at the door in three sentences. What is wearing, add face details. Describe the person at the door in three sentences. What is wearing, add face details."
                    font.pixelSize: 18

                    anchors.fill: parent
                    anchors.margins: 12
                    wrapMode: Label.Wrap
                }
            }
        }

    }
}
