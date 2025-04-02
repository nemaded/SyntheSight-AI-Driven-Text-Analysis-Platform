import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs';
import { Face } from 'kalidokit';

// Initialize face detector
let detector = null;

// Initialize the face detector model
export const initFaceDetector = async () => {
  if (detector) return detector;
  
  await tf.setBackend('webgl');
  
  detector = await faceLandmarksDetection.createDetector(
    faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
    {
      runtime: 'tfjs',
      refineLandmarks: true,
      maxFaces: 1
    }
  );
  
  return detector;
};

// Process video frame to get face data
export const processFrame = async (video) => {
  if (!detector) {
    await initFaceDetector();
  }
  
  // Get face landmarks
  const faces = await detector.estimateFaces(video);
  
  if (faces.length === 0) {
    return null;
  }
  
  // Get the first face
  const face = faces[0];
  
  // Convert landmarks to kalidokit format
  const facePose = Face.solve(face.keypoints, {
    runtime: 'tfjs',
    video: video
  });
  
  return facePose;
};