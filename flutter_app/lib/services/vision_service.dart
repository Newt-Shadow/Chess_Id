import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';

class VisionService {
  late CameraController controller;
  late FlutterVision vision;
  
  Future<void> init() async {
    vision = FlutterVision();
    final cams = await availableCameras();
    controller = CameraController(cams[0], ResolutionPreset.high, enableAudio: false);
    await controller.initialize();
    await vision.loadYoloModel(
      modelPath: 'assets/models/best_float16.tflite',
      labels: 'assets/models/labels.txt',
      modelVersion: "yolov8",
      quantization: true,
      numThreads: 4,
      useGpu: true,
    );
  }

  void stream(Function(List<Map<String, dynamic>>) cb) {
    controller.startImageStream((img) async {
      final res = await vision.yoloOnFrame(
        bytesList: img.planes.map((p)=>p.bytes).toList(),
        imageHeight: img.height, imageWidth: img.width,
        iouThreshold: 0.5, confThreshold: 0.5
      );
      cb(res);
    });
  }
}