// VoiceHelper.js - Utility functions for voice management

/**
 * Finds the best voice based on preference
 * @param {Array} voices - Available speech synthesis voices
 * @param {String} preference - Voice preference ('female', 'male', or 'default')
 * @returns {Object} The selected voice object
 */
export const findBestVoice = (voices, preference = 'default') => {
    if (!voices || voices.length === 0) return null;
    
    let selectedVoice = null;
    
    // Try to find preferred voice type
    if (preference === 'female') {
      selectedVoice = voices.find(voice => 
        voice.name.toLowerCase().includes('female') || 
        voice.name.toLowerCase().includes('woman') ||
        voice.name.toLowerCase().includes('girl')
      );
    } else if (preference === 'male') {
      selectedVoice = voices.find(voice => 
        voice.name.toLowerCase().includes('male') || 
        voice.name.toLowerCase().includes('man') ||
        voice.name.toLowerCase().includes('boy')
      );
    }
    
    // Find best quality voice if preference not found
    if (!selectedVoice) {
      // Try Google voices first (usually higher quality)
      selectedVoice = voices.find(voice => voice.name.includes('Google'));
      
      // Then try Microsoft voices
      if (!selectedVoice) {
        selectedVoice = voices.find(voice => voice.name.includes('Microsoft'));
      }
      
      // Fallback to first voice
      if (!selectedVoice) {
        selectedVoice = voices[0];
      }
    }
    
    return selectedVoice;
  };
  
  /**
   * Add natural pauses to text for more realistic speech
   * @param {String} text - The text to enhance
   * @returns {String} Text with SSML-like pause tags
   */
  export const enhanceTextForSpeech = (text) => {
    if (!text) return '';
    
    // Add pauses after punctuation
    return text
      .replace(/\.\s+/g, '. <break time="400ms"/>')
      .replace(/\!\s+/g, '! <break time="500ms"/>')
      .replace(/\?\s+/g, '? <break time="400ms"/>')
      .replace(/\,\s+/g, ', <break time="200ms"/>');
  };
  
  /**
   * Map summarization style to avatar emotion
   * @param {String} style - Summarization style
   * @returns {String} Corresponding emotion
   */
  export const styleToEmotion = (style) => {
    const emotionMap = {
      'default': 'neutral',
      'academic': 'neutral',
      'concise': 'neutral',
      'friendly': 'happy',
      'professional': 'neutral',
      'enthusiastic': 'happy',
      'critical': 'surprised',
      'explanatory': 'neutral',
      'narrative': 'happy',
      'analytical': 'neutral'
    };
    
    return emotionMap[style] || 'neutral';
  };