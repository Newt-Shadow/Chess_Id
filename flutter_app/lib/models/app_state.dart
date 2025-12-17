import 'package:flutter/material.dart';
import 'package:dartchess/dartchess.dart' as chess;
import 'package:stockfish/stockfish.dart';

enum GameStatus { calibration, playing, illegal }

class AppState extends ChangeNotifier {
  chess.Position _game = chess.Chess.initial;
  final List<String> _history = [];
  GameStatus _status = GameStatus.calibration;
  
  // --- STABILITY ENGINE ---
  Map<int, String> _lastVisionBoard = {};
  int _stabilityCounter = 0;
  static const int _stabilityThreshold = 15; // Require 15 frames (~0.5s) of same board to confirm
  
  late Stockfish _engine;
  final ValueNotifier<double> eval = ValueNotifier(0.0);

  AppState() {
    _engine = Stockfish();
    _engine.stdout.listen((line) {
      if (line.contains('cp ')) {
        final score = int.tryParse(line.split('cp ')[1].split(' ')[0]);
        if (score != null) eval.value = score / 100.0;
      }
    });
  }

  // Getters
  chess.Position get game => _game;
  List<String> get history => _history;
  GameStatus get status => _status;
  String? get lastMoveSan => _history.isEmpty ? null : _history.last;

  void setStatus(GameStatus s) { 
    _status = s; 
    notifyListeners(); 
  }

  // --- CORE LOGIC: Vision to Chess Move ---
  void processVision(Map<int, String> rawVision) {
    if (_status != GameStatus.playing) return;

    // 1. Stability Check: Ignore flickering/hands moving
    if (_mapsEqual(rawVision, _lastVisionBoard)) {
      _stabilityCounter++;
    } else {
      _stabilityCounter = 0;
      _lastVisionBoard = Map.from(rawVision);
      return; // Board is changing (hand is moving), wait
    }

    if (_stabilityCounter < _stabilityThreshold) return; // Not stable yet

    // 2. Logic: Compare Stable Vision vs Internal Game State
    // We try to find a single move that transforms Internal State -> Vision State
    _findAndApplyMove(rawVision);
  }

  void _findAndApplyMove(Map<int, String> visionBoard) {
    // Get all legal moves from current position
    final legalMoves = _game.legalMoves;

    for (final move in legalMoves) {
      // Simulate the move
      final virtualBoard = _game.play(move);
      
      // Check if simulated board matches vision board
      if (_matchesVision(virtualBoard, visionBoard)) {
        // FOUND IT! Apply to real game
        _applyMove(move);
        return;
      }
    }
  }

  bool _matchesVision(chess.Position pos, Map<int, String> vision) {
    int matchCount = 0;
    int mismatchCount = 0;

    for (int i = 0; i < 64; i++) {
      final piece = pos.board[i];
      final visionPiece = vision[i];

      // Logic: If vision sees a piece, it MUST match the game piece
      // If vision sees nothing, we assume empty (unless occluded, but we trust stability)
      if (visionPiece != null) {
        // Convert internal 'wP' to vision 'wP' format if needed
        // Assuming YOLO classes are 'wP', 'bK' etc. matching dartchess
        final internalTag = piece?.type.name.toLowerCase(); // p, k, q
        final internalColor = piece?.color == chess.Side.white ? 'w' : 'b';
        final fullTag = piece != null ? "$internalColor${internalTag!.toUpperCase()}" : null; 
        
        // Note: YOLO classes might be 'wP', dartchess might use different notation.
        // Ensure your YOLO classes (labels.txt) match exactly: wP, wK, bQ, etc.
        if (fullTag == visionPiece) {
          matchCount++;
        } else {
          mismatchCount++;
        }
      }
    }
    // High tolerance match: If 90% of seen pieces match, it's the move
    return mismatchCount == 0 && matchCount > 5; 
  }

  void _applyMove(chess.Move<dynamic> move) {
    _game = _game.play(move);
    _history.add(move.san);
    
    // Engine Analysis
    _engine.stdin = 'position fen ${_game.fen}\ngo movetime 500';
    
    // Reset stability to prevent double-move
    _stabilityCounter = 0; 
    notifyListeners();
  }

  bool _mapsEqual(Map a, Map b) {
    if (a.length != b.length) return false;
    for (var key in a.keys) {
      if (b[key] != a[key]) return false;
    }
    return true;
  }
}