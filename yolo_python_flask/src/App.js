import React, { useEffect, useState, useRef } from 'react';
import ThreeARScene from './ThreeARScene';

function App() {
  const [predictions, setPredictions] = useState([]);
  const videoRef = useRef(null);

  // Set up video element.
  useEffect(() => {
    const video = videoRef.current;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => video.play();
      })
      .catch(err => console.error('Error accessing camera:', err));
  }, []);

  // Capture a frame and send it to the detection backend every second.
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!videoRef.current) return;
      const video = videoRef.current;
      // Create a temporary canvas to capture video frame.
      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      const ctx = captureCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      // Convert canvas image to base64.
      const imageData = captureCanvas.toDataURL('image/jpeg');
      
      try {
        const res = await fetch('http://localhost:5000/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData })
        });
        const data = await res.json();
        if (data.boxes) {
          setPredictions(data.boxes);
        }
      } catch (error) {
        console.error("Detection error:", error);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      {/* Webcam Video */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          zIndex: 0,
        }}
      ></video>
      
      {/* Three.js AR Overlay */}
      <ThreeARScene predictions={predictions} />

      {/* Optional: Display Bounding Boxes as HTML */}
      <div style={{
        position: 'absolute', top: 0, left: 0,
        width: '100%', height: '100%', pointerEvents: 'none'
      }}>
        {predictions.map((box, i) => {
          const [x, y, width, height] = box;
          return (
            <div key={i} style={{
              position: 'absolute',
              border: '2px solid red',
              left: `${x}px`,
              top: `${y}px`,
              width: `${width}px`,
              height: `${height}px`,
              boxSizing: 'border-box',
              color: 'red',
              fontWeight: 'bold',
              backgroundColor: 'rgba(255,0,0,0.1)'
            }}>
              {/* Optionally add label here */}
            </div>
          );
        })}
      </div>

      {/* Info Panel */}
      <div style={{
        position: 'absolute',
        top: 10,
        left: 10,
        color: 'white',
        zIndex: 100,
        background: 'rgba(0,0,0,0.5)',
        padding: '5px',
        borderRadius: '4px'
      }}>
        {predictions.length > 0 ? `Detections: ${predictions.length}` : 'No detections'}
      </div>
    </div>
  );
}

export default App;
