import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/asset_loader.dart';
import 'home_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    // 1. Download Assets if needed
    await AssetLoader.init();
    
    // 2. Artificial delay for branding (optional)
    await Future.delayed(const Duration(seconds: 2));

    // 3. Go to Home
    if (mounted) {
      Navigator.pushReplacement(
        context, 
        MaterialPageRoute(builder: (_) => const HomeScreen())
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.downloading, size: 80, color: Colors.greenAccent),
            const SizedBox(height: 20),
            Text("SETTING UP BOARD...", style: GoogleFonts.bebasNeue(fontSize: 30, color: Colors.white)),
            const SizedBox(height: 10),
            const CircularProgressIndicator(color: Colors.greenAccent),
          ],
        ),
      ),
    );
  }
}