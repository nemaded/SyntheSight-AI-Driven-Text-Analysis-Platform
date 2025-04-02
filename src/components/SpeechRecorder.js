import React, { useState, useRef, useEffect } from 'react';

const SpeechRecorder = ({ onTranscriptionComplete }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [finalTranscription, setFinalTranscription] = useState('');
  const [error, setError] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recognitionRef = useRef(null);
  
  // Set up speech recognition
  useEffect(() => {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      
      recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        // Update transcription with both final and interim results
        const fullTranscript = finalTranscript || interimTranscript;
        setTranscription(fullTranscript);
        // Save final transcription to state so we can use it later
        if (finalTranscript) {
          setFinalTranscription(prev => prev + ' ' + finalTranscript);
        }
        setError(null); // Clear any errors when we get speech
      };
      
      recognition.onerror = (event) => {
        console.log('Speech recognition error', event.error);
        
        // Don't show no-speech error (it's too sensitive)
        if (event.error !== 'no-speech') {
          setError(`Error: ${event.error}`);
        }
      };
      
      recognitionRef.current = recognition;
    }
    
    // Cleanup on component unmount
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          // Ignore errors when stopping
        }
      }
    };
  }, []);

  // Start recording
  const startRecording = async () => {
    try {
      // Reset state
      setTranscription('');
      setFinalTranscription('');
      setError(null);
      audioChunksRef.current = [];
      setAudioURL(null);
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Create media recorder for saving audio
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      // Set up event listeners
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        // Create audio blob when recording is done
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);
        
        // Get the final transcription
        const finalText = (finalTranscription || transcription).trim();
        
        // Finalize transcription and send it to parent immediately
        if (finalText) {
          console.log("Sending transcription to parent:", finalText);
          // Call this IMMEDIATELY - don't wait for state updates
          onTranscriptionComplete(finalText);
          setError(null); // Clear any errors
        } else {
          setError("No speech detected. Please try again.");
        }
        
        setIsProcessing(false);
      };
      
      // Start audio recording
      mediaRecorder.start();
      
      // Start speech recognition
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start();
        } catch (e) {
          // If already started, stop and restart
          if (e.message === 'Failed to execute \'start\' on \'SpeechRecognition\': recognition has already started.') {
            recognitionRef.current.stop();
            setTimeout(() => {
              recognitionRef.current.start();
            }, 100);
          } else {
            throw e;
          }
        }
      } else {
        setError("Speech recognition is not supported in this browser.");
      }
      
      setIsRecording(true);
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Could not access microphone. Please check your browser permissions.');
      setIsProcessing(false);
    }
  };

  // Stop recording
  const stopRecording = () => {
    setIsProcessing(true);
    
    // Stop speech recognition
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.log('Error stopping recognition', e);
      }
    }
    
    // Stop media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    
    setIsRecording(false);
  };

  // Play the recorded audio
  const playAudio = () => {
    if (audioURL) {
      const audio = new Audio(audioURL);
      audio.play();
    }
  };

  // Manual apply transcription function to directly call parent
  const applyTranscription = () => {
    const finalText = (finalTranscription || transcription).trim();
    if (finalText) {
      console.log("Manually applying transcription:", finalText);
      onTranscriptionComplete(finalText);
    }
  };

  return (
    <div className="speech-recorder">
      <div className="recorder-controls">
        {!isRecording && !isProcessing ? (
          <button 
            type="button"
            className="record-button" 
            onClick={startRecording}
          >
            <i className="fas fa-microphone"></i> Start Recording
          </button>
        ) : isRecording ? (
          <button 
            type="button"
            className="stop-button" 
            onClick={stopRecording}
          >
            <i className="fas fa-stop"></i> Stop Recording
          </button>
        ) : (
          <button 
            type="button"
            className="processing-button" 
            disabled
          >
            <i className="fas fa-spinner fa-spin"></i> Processing...
          </button>
        )}
      </div>
      
      {isRecording && (
        <div className="recording-indicator">
          <div className="pulse-animation"></div>
          <p>Recording... Speak clearly into your microphone</p>
          {transcription && (
            <div className="live-transcription">
              {transcription}
            </div>
          )}
        </div>
      )}
      
      {!isRecording && (finalTranscription || transcription) && (
        <div className="transcription-box">
          <h4>Transcription:</h4>
          <p>{finalTranscription || transcription}</p>
          <div className="transcription-actions">
            {audioURL && (
              <button 
                type="button" 
                className="secondary-button play-button"
                onClick={playAudio}
              >
                <i className="fas fa-play"></i> Play Recording
              </button>
            )}
            <button 
              type="button" 
              className="primary-button apply-button"
              onClick={applyTranscription}
            >
              <i className="fas fa-check"></i> Use This Text
            </button>
          </div>
        </div>
      )}
      
      {error && !(finalTranscription || transcription) && (
        <div className="error-message">
          {error}
        </div>
      )}
    </div>
  );
};

export default SpeechRecorder;