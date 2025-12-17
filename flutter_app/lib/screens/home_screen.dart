import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'game_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF262421), Color(0xFF1b1a19)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.videogame_asset, size: 80, color: Colors.white54),
              const SizedBox(height: 20),
              Text(
                "CHESS VISION",
                style: GoogleFonts.bebasNeue(fontSize: 60, color: Colors.white, letterSpacing: 5),
              ),
              Text(
                "AI-Powered Game Recorder",
                style: GoogleFonts.robotoMono(fontSize: 16, color: Colors.white54),
              ),
              const SizedBox(height: 60),
              ElevatedButton.icon(
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const GameScreen()));
                },
                icon: const Icon(Icons.play_arrow, size: 30),
                label: const Text("NEW GAME", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF81B64C),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                  elevation: 10,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}