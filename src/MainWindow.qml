import QtQuick 2.0
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

    required property QtMainWindow ctx

    property int img_id: 1
    property int img_max_id: 2

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

            padding: 8

            Image {
                id: data_img
                source: "images/dummy.png"
                width: 960
                height: 540
                fillMode: Image.PreserveAspectCrop
                antialiasing: false
                anchors.fill: parent

                Button {
                    text: "Load image"
                    font.weight: Font.Medium
                    font.pixelSize: 18
                    Material.roundedScale: Material.SmallScale
                    height: 60

                    onClicked: {
                        if (!ctx.processing) {

                            var image_path = "images/img0" + img_id + ".jpg"
                            console.log(image_path)
                            data_img.source = image_path
                            ctx.loadImage(String("../../" + image_path))
                            img_id++
                            if (img_id > img_max_id) {
                                img_id = 1
                            }
                        } else {
                            console.log("Processing image")
                        }
                    }
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
                height: parent.height * 0.5

                Label {
                    text: root.ctx.prompt
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
                height: parent.height * 0.5

                Label {
                    id: agent_resp
                    text: root.ctx.response
                    font.pixelSize: 18

                    anchors.fill: parent
                    anchors.margins: 12
                    wrapMode: Label.Wrap
                }
            }
        }

    }
}
