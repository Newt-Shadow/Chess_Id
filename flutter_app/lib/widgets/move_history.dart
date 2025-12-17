import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/app_state.dart';

class MoveHistory extends StatelessWidget {
  const MoveHistory({super.key});

  @override
  Widget build(BuildContext context) {
    final history = context.watch<AppState>().history;
    final ScrollController scrollController = ScrollController();

    // Auto-scroll to bottom
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (scrollController.hasClients) {
        scrollController.jumpTo(scrollController.position.maxScrollExtent);
      }
    });

    return Container(
      color: const Color(0xFF21201D),
      child: ListView.builder(
        controller: scrollController,
        padding: const EdgeInsets.all(10),
        itemCount: (history.length / 2).ceil(),
        itemBuilder: (ctx, i) {
          final moveNum = i + 1;
          final whiteMove = history[i * 2];
          final blackMove = (i * 2 + 1 < history.length) ? history[i * 2 + 1] : "";

          return Container(
            padding: const EdgeInsets.symmetric(vertical: 4),
            decoration: BoxDecoration(
              color: i % 2 == 0 ? Colors.transparent : Colors.white.withOpacity(0.05),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 40,
                  child: Text("$moveNum.", style: GoogleFonts.robotoMono(color: Colors.grey, fontWeight: FontWeight.bold)),
                ),
                Expanded(
                  child: Text(whiteMove, style: GoogleFonts.roboto(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w500)),
                ),
                Expanded(
                  child: Text(blackMove, style: GoogleFonts.roboto(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w500)),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}