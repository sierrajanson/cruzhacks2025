// src/Camera.js
import React, { useState, useRef, useEffect } from 'react';
import ThreeARScene from './ThreeARScene';

const Camera = () => {
  const [stream, setStream] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  // Start the camera
  useEffect(() => {
    const startCamera = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' }
        });
        setStream(mediaStream);
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };
    
    startCamera();
    
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  // Process frames and send to server
  useEffect(() => {
    if (!stream || !videoRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    let animationFrameId;
    let lastDetectionTime = 0;
    const DETECTION_INTERVAL = 200; // milliseconds
    
    const processFrame = async (timestamp) => {
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw the current video frame to the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Only send frames for detection at the specified interval
        if (timestamp - lastDetectionTime > DETECTION_INTERVAL) {
          lastDetectionTime = timestamp;
          
          try {
            // Convert canvas to base64 image
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to server for object detection
            const response = await fetch('http://localhost:5000/detect', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image: imageData })
            });
            
            if (response.ok) {
              const data = await response.json();
              if (data.boxes) {
                setPredictions(data.boxes);
              }
            }
          } catch (error) {
            console.error('Error sending frame to server:', error);
          }
        }
      }
      
      animationFrameId = requestAnimationFrame(processFrame);
    };
    
    // Start processing frames
    animationFrameId = requestAnimationFrame(processFrame);
    
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [stream]);
  
  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      {/* Hidden video element to receive camera stream */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ position: 'absolute', width: '100%', height: '100%', objectFit: 'cover' }}
      />
      
      {/* Hidden canvas for processing frames */}
      <canvas
        ref={canvasRef}
        style={{ display: 'none' }}
      />
      
      {/* Three.js scene overlay */}
      <ThreeARScene predictions={predictions} />
    </div>
  );
};

export default Camera;