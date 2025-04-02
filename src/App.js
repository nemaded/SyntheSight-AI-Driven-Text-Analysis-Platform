import React, { useState, useEffect } from 'react';
import './App.css';
import TalkingAvatar from './components/TalkingAvatar';
import SummaryResult from './components/SummaryResult';
import SpeechRecorder from './components/SpeechRecorder';
import ParticleBackground from './components/ParticleBackground';
import ApplicationFlowTabs from './components/ApplicationFlowTabs';

function App() {
  // State management
  const [showIntro, setShowIntro] = useState(true);
  const [step, setStep] = useState(1);
  const [inputType, setInputType] = useState(null);
  const [summarizationType, setSummarizationType] = useState(null);
  const [styles, setStyles] = useState([]);
  const [selectedStyle, setSelectedStyle] = useState('default');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [minLength, setMinLength] = useState(30);
  const [maxLength, setMaxLength] = useState(150);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Function to start the application
  const startApplication = () => {
    setShowIntro(false);
  };
  
  // Hard-coded summarization types
  const hardcodedSummarizationTypes = [
    {
      name: "abstractive",
      description: "AI-generated summary that paraphrases the content"
    },
    {
      name: "extractive",
      description: "Selects and combines the most important sentences from the original text"
    }
  ];

  // Hard-coded styles for each type
  const hardcodedStyles = {
    abstractive: [
      { name: 'default', description: 'Balanced summary with key information' },
      { name: 'concise', description: 'Very brief summary focusing only on the most critical points' },
      { name: 'detailed', description: 'Comprehensive summary covering more information' },
      { name: 'very_detailed', description: 'Highly comprehensive summary with extensive details' },
      { name: 'aggressive', description: 'Highly abstractive summary that condenses information significantly' },
      { name: 'creative', description: 'More paraphrased and creatively reworded summary' },
      { name: 'bullets', description: 'Summary formatted as bullet points' },
      { name: 'eli5', description: 'Explain Like I\'m 5 - Summary in simple language' },
      { name: 'academic', description: 'Formal academic style summary' }
    ],
    extractive: [
      { name: 'tfidf_basic', description: 'Basic TF-IDF extractive summary highlighting key sentences' },
      { name: 'tfidf_short', description: 'Very concise TF-IDF summary with only the most critical sentences' },
      { name: 'tfidf_detailed', description: 'Comprehensive TF-IDF extractive summary with more context' },
      { name: 'textrank', description: 'Graph-based extractive summary using TextRank algorithm' },
      { name: 'centroid', description: 'Centroid-based extractive summary focusing on central concepts' }
    ]
  };

  // Base API URL - replace with your API URL
  const API_URL = 'https://b9c0-35-247-90-209.ngrok-free.app';

  // Update styles when summarization type changes
  useEffect(() => {
    if (summarizationType) {
      // Use hard-coded styles instead of fetching
      setStyles(hardcodedStyles[summarizationType] || []);
      
      // Set default style based on the summarization type
      if (summarizationType === 'abstractive') {
        setSelectedStyle('default');
      } else {
        setSelectedStyle('tfidf_basic');
      }
    }
  }, [summarizationType]);

  // Handle step navigation
  const nextStep = () => {
    setStep(step + 1);
  };

  const prevStep = () => {
    setStep(step - 1);
  };

  // Handle input type selection
  const handleInputTypeSelect = (type) => {
    setInputType(type);
    nextStep();
  };

  // Handle summarization type selection
  const handleSummarizationTypeSelect = (type) => {
    setSummarizationType(type);
    nextStep();
  };

  // Handle style selection
  const handleStyleSelect = (style) => {
    setSelectedStyle(style);
    nextStep();
  };

  // Handle file change
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };
  
  // Handle speech transcription complete
  const handleTranscriptionComplete = (transcription) => {
    console.log("Received transcription in App:", transcription);
    setText(transcription);
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSummary(null);

    // Combine summarization type and style
    const combinedStyle = `${summarizationType}:${selectedStyle}`;
    console.log("Submitting with style:", combinedStyle); // Debug log

    try {
      let response;
      
      if (inputType === 'text' || inputType === 'speech') {
        console.log("Sending text request"); // Debug log
        response = await fetch(`${API_URL}/summarize/text`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text,
            min_length: parseInt(minLength),
            max_length: parseInt(maxLength),
            style: combinedStyle,
          }),
        });
      } 
      else if (inputType === 'url') {
        console.log("Sending URL request"); // Debug log
        response = await fetch(`${API_URL}/summarize/url`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            url,
            min_length: parseInt(minLength),
            max_length: parseInt(maxLength),
            style: combinedStyle,
          }),
        });
      } 
      else if (inputType === 'pdf') {
        console.log("Sending PDF request"); // Debug log
        const formData = new FormData();
        formData.append('file', file);
        formData.append('min_length', minLength);
        formData.append('max_length', maxLength);
        formData.append('style', combinedStyle);

        response = await fetch(`${API_URL}/summarize/pdf`, {
          method: 'POST',
          body: formData,
        });
      }

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to generate summary');
        } else {
          throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
      }

      // Try to parse response as JSON
      try {
        const data = await response.json();
        console.log("Received summary:", data); // Debug log
        setSummary(data);
        nextStep();
      } catch (jsonError) {
        throw new Error("Invalid response format from server");
      }
    } catch (error) {
      console.error("Submission error:", error); // Debug log
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Restart the process
  const handleRestart = () => {
    setStep(1);
    setInputType(null);
    setSummarizationType(null);
    setSelectedStyle('default');
    setText('');
    setUrl('');
    setFile(null);
    setMinLength(30);
    setMaxLength(150);
    setSummary(null);
    setError(null);
  };

  // If showIntro is true, show the introduction page with the application flow tabs
  if (showIntro) {
    return (
      <div className="App">
        <ParticleBackground />
        <header className="App-header">
          <h1>AI Text Summarization - Group 4</h1>
        </header>
        
        <main className="App-main">
          <div className="intro-container">
            <h2>Welcome to AI Text Summarization</h2>
            <p>This tool helps you quickly condense long documents, articles, and other content into concise, readable summaries.</p>
            
            <ApplicationFlowTabs />
            
            <div className="intro-actions">
              <button 
                className="primary-button start-button" 
                onClick={startApplication}
              >
                <i className="fas fa-play-circle"></i> Start Summarizing
              </button>
            </div>
          </div>
        </main>
        
        <footer className="App-footer">
          <p>Text Summarization API powered by PyTorch and Hugging Face   
          </p>
          <p>Developed by Ajit Sarkaar,  Yiming Liu,  Darshan Nemade,  Luyun Zheng,   Deekshitha Gandu</p>

        </footer>
      </div>
    );
  }

  // Render step content
  const renderStepContent = () => {
    switch (step) {
      case 1:
        return (
          <div className="step-content">
            <h2>Select Input Type</h2>
            <p>Choose the type of content you want to summarize:</p>
            <div className="input-type-container">
              <div 
                className="input-type-card" 
                onClick={() => handleInputTypeSelect('text')}
              >
                <div className="input-icon">
                  <i className="fas fa-file-alt"></i>
                </div>
                <h3>Text</h3>
                <p>Paste your text directly</p>
              </div>
              <div 
                className="input-type-card" 
                onClick={() => handleInputTypeSelect('speech')}
              >
                <div className="input-icon">
                  <i className="fas fa-microphone"></i>
                </div>
                <h3>Speech</h3>
                <p>Record and summarize your voice</p>
              </div>
              <div 
                className="input-type-card" 
                onClick={() => handleInputTypeSelect('url')}
              >
                <div className="input-icon">
                  <i className="fas fa-link"></i>
                </div>
                <h3>URL</h3>
                <p>Summarize web content</p>
              </div>
              <div 
                className="input-type-card" 
                onClick={() => handleInputTypeSelect('pdf')}
              >
                <div className="input-icon">
                  <i className="fas fa-file-pdf"></i>
                </div>
                <h3>PDF</h3>
                <p>Upload a PDF document</p>
              </div>
            </div>
          </div>
        );
      
      case 2:
        return (
          <div className="step-content">
            <h2>Select Summarization Type</h2>
            <p>Choose how you want your content to be processed:</p>
            <div className="input-type-container">
              {hardcodedSummarizationTypes.map((type) => (
                <div 
                  key={type.name}
                  className="input-type-card" 
                  onClick={() => handleSummarizationTypeSelect(type.name)}
                >
                  <div className="input-icon">
                    <i className={type.name === 'abstractive' ? 'fas fa-brain' : 'fas fa-scissors'}></i>
                  </div>
                  <h3>{type.name.charAt(0).toUpperCase() + type.name.slice(1)}</h3>
                  <p>{type.description}</p>
                </div>
              ))}
            </div>
            <div className="navigation-buttons">
              <button onClick={prevStep} className="secondary-button">
                Back
              </button>
            </div>
          </div>
        );
      
      case 3:
        return (
          <div className="step-content">
            <h2>Select Summarization Style</h2>
            <p>Choose the specific style for your {summarizationType} summary:</p>
            <div className="styles-container">
              {styles.map((style) => (
                <div 
                  key={style.name}
                  className={`style-card ${selectedStyle === style.name ? 'selected' : ''}`}
                  onClick={() => handleStyleSelect(style.name)}
                >
                  <h3>{style.name.replace(/_/g, ' ')}</h3>
                  <p>{style.description}</p>
                </div>
              ))}
            </div>
            <div className="navigation-buttons">
              <button onClick={prevStep} className="secondary-button">
                Back
              </button>
            </div>
          </div>
        );
      
      case 4:
        return (
          <div className="step-content">
            <h2>Enter Content to Summarize</h2>
            <form onSubmit={handleSubmit}>
              {inputType === 'text' && (
                <div className="form-group">
                  <label htmlFor="text">Enter your text:</label>
                  <textarea
                    id="text"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows="10"
                    required
                    placeholder="Paste your text here..."
                  />
                </div>
              )}
              
              {inputType === 'speech' && (
                  <div className="form-group">
                    <label>Record your speech:</label>
                    <SpeechRecorder
                      onTranscriptionComplete={handleTranscriptionComplete}
                    />
                    {text && (
                      <div className="speech-preview">
                        <h4>Transcribed Text:</h4>
                        <p>{text}</p>
                        <button 
                          type="button" 
                          onClick={() => setText('')} 
                          className="secondary-button"
                          style={{ marginTop: '0.5rem' }}
                        >
                          Clear Text
                        </button>
                      </div>
                    )}
                  </div>
                )}
              
              {inputType === 'url' && (
                <div className="form-group">
                  <label htmlFor="url">Enter URL:</label>
                  <input
                    type="url"
                    id="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    required
                    placeholder="https://example.com/article"
                  />
                </div>
              )}
              
              {inputType === 'pdf' && (
                <div className="form-group">
                  <label htmlFor="file">Upload PDF:</label>
                  <div className="file-input-container">
                    <input
                      type="file"
                      id="file"
                      onChange={handleFileChange}
                      accept=".pdf"
                      required
                    />
                    {file && (
                      <div className="file-info">
                        <p>{file.name}</p>
                        <span>{(file.size / 1024).toFixed(2)} KB</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              <div className="form-group length-controls">
                <div>
                  <label htmlFor="minLength">Minimum Length:</label>
                  <input
                    type="number"
                    id="minLength"
                    value={minLength}
                    onChange={(e) => setMinLength(e.target.value)}
                    min="10"
                    max="200"
                  />
                </div>
                
                <div>
                  <label htmlFor="maxLength">Maximum Length:</label>
                  <input
                    type="number"
                    id="maxLength"
                    value={maxLength}
                    onChange={(e) => setMaxLength(e.target.value)}
                    min="50"
                    max="500"
                  />
                </div>
              </div>
              
              <div className="form-group">
                <div className="summary-info">
                  <p><strong>Input Type:</strong> {inputType}</p>
                  <p><strong>Summarization Type:</strong> {summarizationType}</p>
                  <p><strong>Style:</strong> {selectedStyle.replace(/_/g, ' ')}</p>
                </div>
              </div>
              
              <div className="navigation-buttons">
                <button type="button" onClick={prevStep} className="secondary-button">
                  Back
                </button>
                <button type="submit" className="primary-button" disabled={loading || (inputType === 'speech' && !text)}>
                  {loading ? 'Summarizing...' : 'Generate Summary'}
                </button>
              </div>
            </form>
            
            {error && <div className="error-message">{error}</div>}
          </div>
        );
      
      case 5:
        return (
          <div className="step-content">
            <h2>Summary Results</h2>
            {summary && (
              <SummaryResult summary={summary} summarizationType={summarizationType} />
            )}
            
            <div className="navigation-buttons">
              <button onClick={prevStep} className="secondary-button">
                Back
              </button>
              <button onClick={handleRestart} className="primary-button">
                Start New Summary
              </button>
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="App">
      <ParticleBackground />
      <header className="App-header">
        <h1>AI Text Summarization -  Group 4</h1>
      </header>
      
      <main className="App-main">
        <div className="progress-bar">
          <div className={`progress-step ${step >= 1 ? 'active' : ''}`}>
            <div className="step-number">1</div>
            <div className="step-label">Input Type</div>
          </div>
          <div className={`progress-step ${step >= 2 ? 'active' : ''}`}>
            <div className="step-number">2</div>
            <div className="step-label">Method</div>
          </div>
          <div className={`progress-step ${step >= 3 ? 'active' : ''}`}>
            <div className="step-number">3</div>
            <div className="step-label">Style</div>
          </div>
          <div className={`progress-step ${step >= 4 ? 'active' : ''}`}>
            <div className="step-number">4</div>
            <div className="step-label">Content</div>
          </div>
          <div className={`progress-step ${step >= 5 ? 'active' : ''}`}>
            <div className="step-number">5</div>
            <div className="step-label">Summary</div>
          </div>
        </div>
        
        {renderStepContent()}
      </main>
      
      <footer className="App-footer">
        <p>Text Summarization API powered by PyTorch and Hugging Face</p>
      </footer>
    </div>
  );
}

export default App;