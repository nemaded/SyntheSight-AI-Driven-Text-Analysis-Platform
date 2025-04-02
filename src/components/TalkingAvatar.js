import React, { useState, useEffect } from 'react';

// Common languages with their codes
const LANGUAGES = [
  { code: 'en-US', name: 'English (US)' },
  { code: 'es-ES', name: 'Spanish' },
  { code: 'fr-FR', name: 'French' },
  { code: 'de-DE', name: 'German' },
  { code: 'it-IT', name: 'Italian' },
  { code: 'pt-BR', name: 'Portuguese' },
  { code: 'ru-RU', name: 'Russian' },
  { code: 'zh-CN', name: 'Chinese' },
  { code: 'ja-JP', name: 'Japanese' },
  { code: 'ko-KR', name: 'Korean' },
  { code: 'ar-SA', name: 'Arabic' },
  { code: 'hi-IN', name: 'Hindi' },
  { code: 'nl-NL', name: 'Dutch' },
  { code: 'pl-PL', name: 'Polish' },
  { code: 'sv-SE', name: 'Swedish' },
  { code: 'tr-TR', name: 'Turkish' }
];

const TalkingAvatar = ({ text, autoPlay = false }) => {
  const [speaking, setSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('en-US');
  const [filteredVoices, setFilteredVoices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [translating, setTranslating] = useState(false);
  const [translatedText, setTranslatedText] = useState('');
  const [error, setError] = useState(null);

  // Load and filter available voices
  useEffect(() => {
    function loadVoices() {
      const availableVoices = window.speechSynthesis.getVoices();
      if (availableVoices.length > 0) {
        setVoices(availableVoices);
        setLoading(false);
        
        // Filter voices by selected language
        filterVoicesByLanguage(availableVoices, selectedLanguage);
      }
    }
    
    loadVoices();
    
    // Chrome loads voices asynchronously
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    return () => {
      // Clean up
      if (speaking) {
        window.speechSynthesis.cancel();
      }
    };
  }, [speaking]);
  
  // Filter voices when language changes
  useEffect(() => {
    filterVoicesByLanguage(voices, selectedLanguage);
    // Reset translated text when language changes
    setTranslatedText('');
  }, [selectedLanguage, voices]);
  
  const filterVoicesByLanguage = (allVoices, langCode) => {
    // Filter voices by language code
    const matchingVoices = allVoices.filter(voice => 
      voice.lang.toLowerCase().includes(langCode.split('-')[0].toLowerCase())
    );
    
    setFilteredVoices(matchingVoices);
    
    // Select the first matching voice or keep the current one if it matches
    if (matchingVoices.length > 0) {
      if (!selectedVoice || !matchingVoices.some(v => v.name === selectedVoice.name)) {
        setSelectedVoice(matchingVoices[0]);
      }
    } else {
      // If no matching voices, reset to null
      setSelectedVoice(null);
    }
  };

  // Translate text to selected language
  const translateText = async (targetLanguage) => {
    setTranslating(true);
    setError(null);
    
    try {
      // Get the language code without the region specifier
      const langCode = targetLanguage.split('-')[0];
      
      // Don't translate if the selected language is English
      if (langCode === 'en') {
        setTranslatedText(text);
        setTranslating(false);
        return;
      }
      
      // Use the backend API to translate
      const response = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          target_language: langCode
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Translation failed');
      }
      
      const data = await response.json();
      setTranslatedText(data.translated_text);
    } catch (error) {
      console.error('Translation error:', error);
      setError('Failed to translate text. Using original text instead.');
      // Fallback to original text if translation fails
      setTranslatedText(text);
    } finally {
      setTranslating(false);
    }
  };

  // Handle speak button click
  const handleSpeak = async () => {
    if (speaking) {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      setSpeaking(false);
      return;
    }
    
    if (!window.speechSynthesis) {
      alert("Your browser doesn't support text-to-speech. Please try another browser.");
      return;
    }
    
    // Translate text if not already translated
    if (!translatedText) {
      await translateText(selectedLanguage);
    }
    
    // Start speaking
    setSpeaking(true);
    
    // Create and configure utterance
    const utterance = new SpeechSynthesisUtterance(translatedText || text);
    
    // Set language and voice if available
    utterance.lang = selectedLanguage;
    
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    
    // Set speech parameters
    utterance.rate = 1.0;
    utterance.pitch = 1.1;
    
    // Add event handlers
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = (event) => {
      console.error('Speech synthesis error:', event);
      setSpeaking(false);
    };
    
    // Speak the text
    window.speechSynthesis.speak(utterance);
  };

  // Handle language change
  const handleLanguageChange = (e) => {
    const newLanguage = e.target.value;
    setSelectedLanguage(newLanguage);
    // Translate text when language changes
    translateText(newLanguage);
  };

  return (
    <div className="avatar-container">
      <div className="avatar-display" style={{ height: '200px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="avatar-placeholder">
          <img 
            src="https://api.dicebear.com/7.x/personas/svg?seed=talking" 
            alt="Avatar" 
            style={{ 
              width: '150px', 
              height: '150px', 
              borderRadius: '50%',
              animation: speaking ? 'pulse 1.5s infinite' : 'none'
            }} 
          />
          
          <style>
            {`
              @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
              }
            `}
          </style>
        </div>
      </div>
      
      <div className="avatar-controls" style={{ marginTop: '15px', width: '100%' }}>
        {/* Language selection */}
        <div className="language-selection" style={{ marginBottom: '10px' }}>
          <label htmlFor="language-select" style={{ marginRight: '10px', fontWeight: '500' }}>Language:</label>
          <select 
            id="language-select"
            value={selectedLanguage}
            onChange={handleLanguageChange}
            style={{ padding: '8px', borderRadius: '4px', width: '100%', marginBottom: '10px' }}
            disabled={speaking}
          >
            {LANGUAGES.map(lang => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>
        
        {/* Voice selection */}
        <div className="voice-selection" style={{ marginBottom: '15px' }}>
          <label htmlFor="voice-select" style={{ marginRight: '10px', fontWeight: '500' }}>Voice:</label>
          <select 
            id="voice-select"
            value={selectedVoice ? selectedVoice.name : ''}
            onChange={(e) => {
              const voice = voices.find(v => v.name === e.target.value);
              setSelectedVoice(voice);
            }}
            style={{ padding: '8px', borderRadius: '4px', width: '100%' }}
            disabled={filteredVoices.length === 0 || speaking}
          >
            {filteredVoices.length > 0 ? (
              filteredVoices.map(voice => (
                <option key={voice.name} value={voice.name}>
                  {voice.name}
                </option>
              ))
            ) : (
              <option value="">No voices available for this language</option>
            )}
          </select>
        </div>
        
        {/* Translation status */}
        {translating && (
          <div style={{ textAlign: 'center', marginBottom: '10px', color: '#666' }}>
            Translating text to {LANGUAGES.find(l => l.code === selectedLanguage)?.name}...
          </div>
        )}
        
        {/* Error message */}
        {error && (
          <div style={{ textAlign: 'center', marginBottom: '10px', color: '#d32f2f' }}>
            {error}
          </div>
        )}
        
        {/* Speak button */}
        <button 
          onClick={handleSpeak}
          className={`avatar-speak-button ${speaking ? 'speaking' : ''}`}
          disabled={loading || filteredVoices.length === 0 || translating}
          style={{ width: '100%' }}
        >
          {loading ? (
            'Loading voices...'
          ) : translating ? (
            'Translating...'
          ) : speaking ? (
            <>
              <i className="fas fa-stop"></i> Stop Speaking
            </>
          ) : (
            <>
              <i className="fas fa-play"></i> Speak in {LANGUAGES.find(l => l.code === selectedLanguage)?.name}
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default TalkingAvatar;