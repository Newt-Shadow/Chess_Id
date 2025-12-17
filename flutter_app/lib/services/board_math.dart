import 'dart:math';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class BoardMath {
  final List<cv.Point2f> _dst = [
    cv.Point2f(0, 0), cv.Point2f(640, 0),
    cv.Point2f(640, 640), cv.Point2f(0, 640)
  ];
  cv.Mat? _matrix;

  void calibrate(List<Point<double>> corners) {
    if (corners.length != 4) return;
    // Sort TL, TR, BR, BL
    corners.sort((a, b) => a.y.compareTo(b.y));
    final top = corners.sublist(0, 2)..sort((a, b) => a.x.compareTo(b.x));
    final bot = corners.sublist(2, 4)..sort((a, b) => a.x.compareTo(b.x));
    final src = [...top, ...bot.reversed].map((p) => cv.Point2f(p.x, p.y)).toList();
    _matrix = cv.getPerspectiveTransform(src, _dst);
  }

  int? getSquare(Point<double> p) {
    if (_matrix == null) return null;
    final srcM = cv.Mat.fromVec(cv.VecPoint2f.fromList([cv.Point2f(p.x, p.y)]));
    final dstM = cv.perspectiveTransform(srcM, _matrix!);
    final x = dstM.at<double>(0, 0);
    final y = dstM.at<double>(0, 1);
    if (x < 0 || x > 640 || y < 0 || y > 640) return null;
    return (y ~/ 80) * 8 + (x ~/ 80);
  }
}