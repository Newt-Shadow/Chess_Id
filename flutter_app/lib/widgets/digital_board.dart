import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:provider/provider.dart';
import 'package:dartchess/dartchess.dart' as chess;
import '../models/app_state.dart';

class DigitalBoard extends StatelessWidget {
  const DigitalBoard({super.key});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final board = state.game.board;
    final lastMove = state.game.history.isEmpty ? null : state.game.history.last.move;

    return AspectRatio(
      aspectRatio: 1.0,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: const Color(0xFF333333), width: 8),
          borderRadius: BorderRadius.circular(4),
          boxShadow: [const BoxShadow(color: Colors.black54, blurRadius: 15, offset: Offset(0, 10))]
        ),
        child: Column(
          children: List.generate(8, (rank) {
            return Expanded(
              child: Row(
                children: List.generate(8, (file) {
                  final squareIndex = (rank * 8) + file;
                  final isLight = (rank + file) % 2 == 0;
                  final piece = board[squareIndex];
                  
                  bool isHighlight = false;
                  if (lastMove != null) {
                    if (lastMove.from == squareIndex || lastMove.to == squareIndex) {
                      isHighlight = true;
                    }
                  }

                  return Expanded(
                    child: Container(
                      color: isHighlight 
                          ? const Color(0xFFBBCB2B).withOpacity(0.8)
                          : isLight ? const Color(0xFFEEEED2) : const Color(0xFF769656),
                      child: piece != null ? _buildPiece(piece) : null,
                    ),
                  );
                }),
              ),
            );
          }),
        ),
      ),
    );
  }

  Widget _buildPiece(chess.Piece piece) {
    final color = piece.color == chess.Side.white ? 'w' : 'b';
    final type = piece.type.name.toUpperCase(); // P, N, B...
    
    // DIRECT ASSET LOAD (Fast & Offline)
    return Padding(
      padding: const EdgeInsets.all(2.0),
      child: SvgPicture.asset("assets/pieces/$color$type.svg"), 
    );
  }
}