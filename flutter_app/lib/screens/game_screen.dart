import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/app_state.dart';
import '../services/vision_service.dart';
import '../services/board_math.dart';
import '../widgets/digital_board.dart';
import '../widgets/move_history.dart';

class GameScreen extends StatefulWidget {
  const GameScreen({super.key});
  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  final VisionService _vis = VisionService();
  final BoardMath _math = BoardMath();
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _vis.init().then((_) {
      setState(() => _ready = true);
      context.read<AppState>().setStatus(GameStatus.calibration);
      _vis.stream(_onFrame);
    });
  }

  void _onFrame(List<Map<String, dynamic>> res) {
    if (!mounted) return;
    final state = context.read<AppState>();
    
    List<Point<double>> corners = [];
    Map<int, String> board = {};

    for (var d in res) {
      final box = d["box"];
      final pt = Point<double>((box[0] + box[2]) / 2, (box[1] + box[3]) / 2);
      if (d["tag"] == "corner") corners.add(pt);
      else if (state.status == GameStatus.playing) {
        final sq = _math.getSquare(pt);
        if (sq != null) board[sq] = d["tag"];
      }
    }

    if (state.status == GameStatus.calibration && corners.length == 4) {
      _math.calibrate(corners);
      Future.delayed(const Duration(seconds: 1), () {
        if(mounted) state.setStatus(GameStatus.playing);
      });
    } else {
      state.processVision(board);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) return const Scaffold(backgroundColor: Colors.black, body: Center(child: CircularProgressIndicator()));
    final state = context.watch<AppState>();

    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        title: Text("LIVE GAME", style: GoogleFonts.oswald(letterSpacing: 2)),
        backgroundColor: const Color(0xFF262421),
        leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: () => Navigator.pop(context)),
        actions: [
          // Indicator for "Recording"
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(color: Colors.red.withOpacity(0.2), borderRadius: BorderRadius.circular(20)),
            child: Row(children: const [
              Icon(Icons.circle, size: 10, color: Colors.red),
              SizedBox(width: 8),
              Text("REC", style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold))
            ]),
          )
        ],
      ),
      body: Column(
        children: [
          // 1. The Board Area
          Expanded(
            flex: 5,
            child: Stack(
              children: [
                // HIDDEN CAMERA (Must exist for stream to work, but we hide it)
                Opacity(opacity: 0.01, child: SizedBox(width: 1, height: 1, child: CameraPreview(_vis.controller))),
                
                // MAIN UI
                Center(
                  child: state.status == GameStatus.calibration 
                    ? _buildCalibrationUI()
                    : const Padding(padding: EdgeInsets.all(16.0), child: DigitalBoard()),
                ),
              ],
            ),
          ),

          // 2. Info & History Area
          Expanded(
            flex: 4,
            child: Column(
              children: [
                // Eval Bar
                LinearProgressIndicator(
                  value: 0.5 + (state.eval.value / 10.0).clamp(-0.5, 0.5),
                  backgroundColor: Colors.grey[800],
                  color: Colors.white,
                  minHeight: 4,
                ),
                // Last Move Big Display
                Container(
                  width: double.infinity,
                  color: const Color(0xFF262421),
                  padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text("LAST MOVE", style: GoogleFonts.roboto(color: Colors.grey, fontSize: 12)),
                      Text(state.lastMoveSan ?? "--", style: GoogleFonts.oswald(color: Colors.greenAccent, fontSize: 32, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
                const Expanded(child: MoveHistory()),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCalibrationUI() {
    return Container(
      color: Colors.black, // Show black bg while calibrating (or show camera if you prefer)
      child: Stack(
        children: [
          Positioned.fill(child: CameraPreview(_vis.controller)), // Show Camera ONLY during calibration
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 300, height: 300,
                  decoration: BoxDecoration(border: Border.all(color: Colors.greenAccent, width: 3)),
                ),
                const SizedBox(height: 20),
                Container(
                  padding: const EdgeInsets.all(12),
                  color: Colors.black54,
                  child: Text("ALIGN BOARD CORNERS", style: GoogleFonts.bebasNeue(color: Colors.white, fontSize: 28)),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _vis.controller.dispose();
    super.dispose();
  }
}