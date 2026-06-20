# Emotion Detection & Music Recommendation System

A web-based emotion detection system that scans your face for 15 seconds, analyzes your emotions, and recommends music based on your detected mood.

## Features

✨ **15-Second Face Scanning** - Captures and analyzes your facial expressions for 15 seconds
📊 **Emotion Percentage Display** - Shows the percentage breakdown of each detected emotion
🎵 **Smart Music Recommendations** - Recommends Spotify songs based on your dominant emotion
🎨 **Beautiful Web Interface** - Modern, responsive UI built with HTML/CSS
📱 **Fully Responsive** - Works on desktop, tablet, and mobile browsers

## Requirements

- Python 3.13+
- Webcam/Camera
- Required Python packages (automatically installed):
  - Flask
  - TensorFlow/Keras
  - OpenCV
  - NumPy

## Installation

1. Navigate to the project directory:
```bash
cd "c:\Users\91897\OneDrive\Desktop\7TH SEM\1 PROJECT MAJOR"
```

2. Install required packages:
```bash
python -m pip install flask tensorflow keras opencv-python numpy
```

## Running the Application

### Method 1: Using the Batch File (Recommended)
Double-click `run_app.bat` to start the application.

### Method 2: Using Python Command
```bash
python app.py
```

### Method 3: Using PowerShell
```powershell
& "C:\Program Files\Python313\python.exe" app.py
```

## How to Use

1. **Start the Server**
   - Run `run_app.bat` or execute `python app.py`
   - Wait for "Running on http://127.0.0.1:5000/" message in terminal

2. **Open the Web Interface**
   - Open your browser and go to: `http://localhost:5000`
   - You should see the Emotion Detector interface

3. **Start Emotion Scanning**
   - Click the "START SCAN" button
   - Position your face in front of the camera
   - Hold still and maintain a natural expression
   - The scan will run for 15 seconds

4. **View Results**
   - After scanning completes, you'll see:
     - Emotion percentages for each emotion type
     - Your dominant emotion highlighted
     - Recommended Spotify songs matching your mood
   - Click "Open on Spotify" to listen to recommended songs

5. **Reset for New Scan**
   - Click "RESET" to clear results and start over

## Project Structure

```
final_project.py          - Original emotion detection script
app.py                    - Flask web server
run_app.bat              - Batch file to run the app
model.h5                 - Pre-trained emotion detection model
haarcascade_frontalface_default.xml - Face detection classifier
templates/
  └── index.html         - Web interface (HTML/CSS/JavaScript)
```

## Emotion Categories

The system detects 7 different emotions:
- 😠 **Angry** - Rock/Metal music recommendations
- 🤮 **Disgust** - Alternative/Rock music
- 😨 **Fear** - Electronic/Intense music
- 😊 **Happy** - Pop/Uplifting music
- 😐 **Neutral** - Lo-Fi/Chill music
- 😢 **Sad** - Ballads/Melancholic music
- 😮 **Surprise** - Upbeat/Energetic music

## API Endpoints

- `GET /` - Main web interface
- `POST /start_scan` - Start 15-second emotion scan
- `POST /stop_scan` - Stop the current scan
- `GET /get_scan_data` - Get current scan results and statistics
- `POST /reset_data` - Reset all data

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'keras'"
**Solution:** Install required packages:
```bash
python -m pip install keras tensorflow opencv-python numpy flask
```

### Issue: "Camera not found" or "Cannot access camera"
**Solution:** 
- Check if your camera is connected and functioning
- Close other applications using the camera
- Check camera permissions in Windows settings

### Issue: "Port 5000 is already in use"
**Solution:** Either:
- Close other applications using port 5000
- Modify the port in `app.py` line: `app.run(debug=True, port=5001)`

### Issue: Emotion detection accuracy is low
**Solution:**
- Ensure good lighting conditions
- Keep your face clearly visible to the camera
- Avoid partial faces or obscured features
- Make clear emotional expressions

## Tips for Better Results

1. **Lighting** - Use bright, natural light or a well-lit room
2. **Distance** - Keep your face about 30-60 cm from the camera
3. **Expressions** - Make clear, distinct emotional expressions
4. **Duration** - The 15-second scan period gives better accuracy with multiple samples
5. **Multiples Scans** - Try multiple scans for more accurate results

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Limitations

- Requires webcam access
- Works best with frontal face views
- May have lower accuracy with:
  - Glasses or sunglasses
  - Face masks
  - Heavy makeup variations
  - Poor lighting conditions
  - Profile views

## Future Enhancements

- [ ] Add facial expression intensity tracking
- [ ] Save scan history and statistics
- [ ] Export reports as PDF
- [ ] Real-time video stream to browser
- [ ] Support for multiple faces
- [ ] Custom music playlist integration
- [ ] Dark mode UI toggle

## License

This project uses:
- OpenCV (BSD License)
- TensorFlow/Keras (Apache 2.0 License)
- Flask (BSD License)

## Support

For issues or suggestions, please review the troubleshooting section or check the console output for error messages.

---

**Enjoy discovering your mood-based music recommendations! 🎵😊**
