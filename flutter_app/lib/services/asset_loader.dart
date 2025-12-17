import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class AssetLoader {
  static const String _baseUrl = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett";
  // Standard piece codes: wP = White Pawn, bK = Black King, etc.
  static const List<String> _pieces = [
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK"
  ];

  static Future<void> init() async {
    final dir = await getApplicationDocumentsDirectory();
    final piecesDir = Directory('${dir.path}/pieces');
    
    if (!await piecesDir.exists()) {
      await piecesDir.create(recursive: true);
    }

    for (String p in _pieces) {
      final file = File('${piecesDir.path}/$p.svg');
      if (!await file.exists()) {
        print("⬇️ Downloading $p.svg...");
        try {
          final res = await http.get(Uri.parse("$_baseUrl/$p.svg"));
          if (res.statusCode == 200) {
            await file.writeAsBytes(res.bodyBytes);
          }
        } catch (e) {
          print("Error downloading $p: $e");
        }
      }
    }
  }

  static Future<String> getPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return "${dir.path}/pieces";
  }
}