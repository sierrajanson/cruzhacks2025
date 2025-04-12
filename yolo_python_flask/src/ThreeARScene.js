// src/ThreeARScene.js
import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

const ThreeARScene = ({ predictions }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  // To store 3D objects for each detection
  const objectsRef = useRef([]);

  useEffect(() => {
    // Set up the Three.js scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    
    // Set up the camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;
    
    // Set up the renderer with clear settings
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true,
      antialias: true
    });
    renderer.setClearColor(0x000000, 0); // Transparent background
    renderer.setSize(window.innerWidth, window.innerHeight);
    rendererRef.current = renderer;
    
    // Add renderer to DOM
    if (mountRef.current) {
      mountRef.current.appendChild(renderer.domElement);
    }
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);
    
    // Add a clearly visible reference object to confirm Three.js is working
    const gridHelper = new THREE.GridHelper(10, 10, 0xff0000, 0x444444);
    scene.add(gridHelper);
    
    const sphereGeometry = new THREE.SphereGeometry(0.3, 32, 32);
    const sphereMaterial = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(0, 2, 0);
    scene.add(sphere);
    
    // Add axes helper
    const axesHelper = new THREE.AxesHelper(2);
    scene.add(axesHelper);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate sphere to confirm animation is working
      if (sphere) {
        sphere.rotation.x += 0.01;
        sphere.rotation.y += 0.01;
      }
      
      renderer.render(scene, camera);
    };
    animate();
    
    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update 3D objects when predictions change
  useEffect(() => {
    if (!sceneRef.current) return;
    
    const scene = sceneRef.current;
    
    // Remove existing objects
    objectsRef.current.forEach(obj => scene.remove(obj));
    objectsRef.current = [];
    
    // Create new visible 3D objects for each prediction
    predictions.forEach((prediction, index) => {
      // Extract prediction data: [x, y, width, height]
      const [x, y, width, height] = prediction;
      
      // Convert screen coordinates to normalized device coordinates
      const ndcX = (x + width/2) / window.innerWidth * 2 - 1;
      const ndcY = -((y + height/2) / window.innerHeight * 2 - 1);
      
      // Create a raycaster to project from camera through the NDC point
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), cameraRef.current);
      
      // Get point at certain distance from camera along ray
      const depth = 3; // Place objects 3 units in front of camera
      const pointAtDepth = new THREE.Vector3();
      raycaster.ray.at(depth, pointAtDepth);
      
      // Create a highly visible 3D object
      const geometry = new THREE.BoxGeometry(width / 200, height / 200, 0.2);
      const material = new THREE.MeshPhongMaterial({
        color: getColorForIndex(index),
        transparent: true,
        opacity: 0.7,
        wireframe: false
      });
      const box = new THREE.Mesh(geometry, material);
      
      // Position the box at the projected point
      box.position.copy(pointAtDepth);
      
      // Make the box always face the camera
      box.lookAt(cameraRef.current.position);
      
      // Add wireframe for better visibility
      const wireframe = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: 0xffffff })
      );
      box.add(wireframe);
      
      // Add text label showing object index for debugging
      const textCanvas = document.createElement('canvas');
      const ctx = textCanvas.getContext('2d');
      textCanvas.width = 128;
      textCanvas.height = 64;
      ctx.fillStyle = 'white';
      ctx.font = '32px Arial';
      ctx.fillText(`Object ${index}`, 10, 40);
      
      const textTexture = new THREE.CanvasTexture(textCanvas);
      const textMaterial = new THREE.MeshBasicMaterial({
        map: textTexture,
        transparent: true
      });
      const textGeometry = new THREE.PlaneGeometry(0.5, 0.25);
      const textMesh = new THREE.Mesh(textGeometry, textMaterial);
      textMesh.position.set(0, geometry.parameters.height / 1.5, 0.11);
      box.add(textMesh);
      
      // Add the box to the scene
      scene.add(box);
      objectsRef.current.push(box);
      
      // Add a connecting line from camera to object for debugging
      const lineMaterial = new THREE.LineBasicMaterial({ color: getColorForIndex(index) });
      const lineGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        pointAtDepth
      ]);
      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);
      objectsRef.current.push(line);
    });
    
  }, [predictions]);

  // Generate a color based on object index
  const getColorForIndex = (index) => {
    const colors = [
      0xff0000, // Red
      0x00ff00, // Green
      0x0000ff, // Blue
      0xffff00, // Yellow
      0xff00ff, // Magenta
      0x00ffff, // Cyan
      0xff8000, // Orange
      0x8000ff  // Purple
    ];
    return colors[index % colors.length];
  };

  return (
    <div
      ref={mountRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 10
      }}
    />
  );
};

export default ThreeARScene;