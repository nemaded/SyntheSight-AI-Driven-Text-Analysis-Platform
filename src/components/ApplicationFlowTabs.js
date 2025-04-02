// components/ApplicationFlowTabs.js
import React, { useState } from 'react';

const ApplicationFlowTabs = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  const tabs = [
    { id: 'overview', label: 'Overview', icon: 'fas fa-info-circle' },
    { id: 'input', label: 'Input', icon: 'fas fa-file-import' },
    { id: 'processing', label: 'Processing', icon: 'fas fa-cogs' },
    { id: 'output', label: 'Output', icon: 'fas fa-file-export' },
    { id: 'styles', label: 'Styles', icon: 'fas fa-paint-brush' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="tab-content-pane">
            <h3>How This Application Works</h3>
            <p>This AI-powered text summarization tool helps you quickly condense long documents, articles, and other content into concise, readable summaries.</p>
            
            <div className="flow-overview">
              <div className="flow-step">
                <div className="flow-icon">
                  <i className="fas fa-file-import"></i>
                </div>
                <div className="flow-text">
                  <h4>1. Input</h4>
                  <p>Provide content through text, URL, PDF, or speech</p>
                </div>
              </div>
              <div className="flow-arrow">→</div>
              <div className="flow-step">
                <div className="flow-icon">
                  <i className="fas fa-brain"></i>
                </div>
                <div className="flow-text">
                  <h4>2. Process</h4>
                  <p>AI analyzes and summarizes content</p>
                </div>
              </div>
              <div className="flow-arrow">→</div>
              <div className="flow-step">
                <div className="flow-icon">
                  <i className="fas fa-file-alt"></i>
                </div>
                <div className="flow-text">
                  <h4>3. Output</h4>
                  <p>View and listen to your tailored summary</p>
                </div>
              </div>
            </div>
            
            <p className="tab-note">Click through the tabs above to learn more about each step of the process.</p>
          </div>
        );
        
      case 'input':
        return (
          <div className="tab-content-pane">
            <h3>Content Input Options</h3>
            <p>Our application offers multiple ways to provide content for summarization:</p>
            
            <div className="input-methods">
              <div className="input-method-card">
                <div className="method-icon">
                  <i className="fas fa-file-alt"></i>
                </div>
                <h4>Text Input</h4>
                <p>Paste any text directly into the application</p>
                <div className="method-note">Best for: Articles, essays, or any text you've already copied</div>
              </div>
              
              <div className="input-method-card">
                <div className="method-icon">
                  <i className="fas fa-microphone"></i>
                </div>
                <h4>Speech Input</h4>
                <p>Record your voice and convert to text automatically</p>
                <div className="method-note">Best for: Meetings, lectures, or hands-free operation</div>
              </div>
              
              <div className="input-method-card">
                <div className="method-icon">
                  <i className="fas fa-link"></i>
                </div>
                <h4>URL Input</h4>
                <p>Enter a web address to summarize online content</p>
                <div className="method-note">Best for: News articles, blog posts, and web pages</div>
              </div>
              
              <div className="input-method-card">
                <div className="method-icon">
                  <i className="fas fa-file-pdf"></i>
                </div>
                <h4>PDF Upload</h4>
                <p>Upload PDF documents for automatic extraction and summarization</p>
                <div className="method-note">Best for: Research papers, reports, and documents</div>
              </div>
            </div>
          </div>
        );
        
        case 'processing':
          return (
            <div className="tab-content-pane">
              <h3>Summarization Methods & Processing Pipeline</h3>
              <p>Our application utilizes state-of-the-art summarization techniques through a sophisticated backend processing pipeline:</p>
              
              <div className="technical-pipeline-container">
                <h4>Backend Architecture</h4>
                <div className="architecture-diagram">
                  <div className="arch-component">
                    <div className="arch-icon"><i className="fas fa-server"></i></div>
                    <div className="arch-label">FastAPI Backend</div>
                  </div>
                  <div className="arch-arrow">→</div>
                  <div className="arch-component">
                    <div className="arch-icon"><i className="fas fa-microchip"></i></div>
                    <div className="arch-label">PyTorch Engine</div>
                  </div>
                  <div className="arch-arrow">→</div>
                  <div className="arch-component">
                    <div className="arch-icon"><i className="fas fa-brain"></i></div>
                    <div className="arch-label">Hugging Face Models</div>
                  </div>
                </div>
                
                <h4 className="processing-section-title">Processing Pipeline Details</h4>
                
                <div className="processing-methods">
                  <div className="processing-method-card abstractive">
                    <div className="method-header">
                      <div className="method-icon">
                        <i className="fas fa-brain"></i>
                      </div>
                      <h4>Abstractive Summarization</h4>
                    </div>
                    <p>Uses neural networks to generate entirely new text that captures the essence of the original content.</p>
                    
                    <div className="technical-details">
                      <h5>Technical Implementation:</h5>
                      <ul className="tech-list">
                        <li><span className="tech-highlight">Model:</span> BART (Bidirectional and Auto-Regressive Transformers)</li>
                        <li><span className="tech-highlight">Architecture:</span> Encoder-decoder transformer neural network</li>
                        <li><span className="tech-highlight">Training:</span> Pre-trained on large text corpora with denoising objectives</li>
                        <li><span className="tech-highlight">Processing Steps:</span> Tokenization → Encoding → Decoding → Beam Search</li>
                      </ul>
                      
                      <div className="method-pipeline detailed">
                        <div className="pipeline-step">
                          <span className="step-badge">1</span>
                          <div className="step-details">
                            <span className="step-name">Text Preprocessing</span>
                            <span className="step-desc">Document cleaning, tokenization into subwords</span>
                          </div>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className="pipeline-step">
                          <span className="step-badge">2</span>
                          <div className="step-details">
                            <span className="step-name">Encoder Processing</span>
                            <span className="step-desc">Bidirectional context embedding of input sequence</span>
                          </div>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className="pipeline-step">
                          <span className="step-badge">3</span>
                          <div className="step-details">
                            <span className="step-name">Decoder Generation</span>
                            <span className="step-desc">Auto-regressive text generation with attention</span>
                          </div>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className="pipeline-step">
                          <span className="step-badge">4</span>
                          <div className="step-details">
                            <span className="step-name">Style Application</span>
                            <span className="step-desc">Parameter adjustment based on selected style</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="code-sample">
                        <pre>
                          <code>
            # Abstractive summarization with BART (simplified)
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,             # Beam search parameter
                no_repeat_ngram_size=3,  # Avoid repetition
                length_penalty=1.0,      # Balance between length and quality
                early_stopping=True      # Stop when all beams reach EOS
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                          </code>
                        </pre>
                      </div>
                    </div>
                    
                    <div className="method-note">Creates completely new text while preserving the original meaning</div>
                  </div>
                  
                  <div className="processing-method-card extractive">
                    <div className="method-header">
                      <div className="method-icon">
                        <i className="fas fa-scissors"></i>
                      </div>
                      <h4>Extractive Summarization</h4>
                    </div>
                    <p>Identifies and extracts the most important sentences from the original text.</p>
                    
                    <div className="technical-details">
                      <h5>Technical Implementation:</h5>
                      <ul className="tech-list">
                        <li><span className="tech-highlight">Algorithms:</span> TF-IDF, TextRank, or Centroid-based approaches</li>
                        <li><span className="tech-highlight">Vectorization:</span> Converts text to numerical representations</li>
                        <li><span className="tech-highlight">Ranking:</span> Sentences scored based on importance metrics</li>
                        <li><span className="tech-highlight">Processing Steps:</span> Segmentation → Vectorization → Scoring → Selection</li>
                      </ul>
                      
                      <div className="method-pipeline detailed">
                        <div className="pipeline-step">
                          <span className="step-badge">1</span>
                          <div className="step-details">
                            <span className="step-name">Sentence Segmentation</span>
                            <span className="step-desc">Splitting text into individual sentences</span>
                          </div>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className="pipeline-step">
                          <span className="step-badge">2</span>
                          <div className="step-details">
                            <span className="step-name">Feature Extraction</span>
                            <span className="step-desc">TF-IDF vectorization of sentences</span>
                          </div>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className="pipeline-step">
                          <span className="step-badge">3</span>
                          <div className="step-details">
                            <span className="step-name">Importance Scoring</span>
                            <span className="step-desc">Sentence ranking by algorithm-specific metrics</span>
                          </div>
                        </div>
                        <div className="pipeline-arrow">→</div>
                        <div className="pipeline-step">
                          <span className="step-badge">4</span>
                          <div className="step-details">
                            <span className="step-name">Sentence Selection</span>
                            <span className="step-desc">Top-ranked sentences assembled in original order</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="code-sample">
                        <pre>
                          <code>
            # Extractive summarization with TF-IDF (simplified)
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = []
            for i in range(len(sentences)):
                score = sum(tfidf_matrix[i].toarray()[0])
                sentence_scores.append((i, score))
            
            # Get top sentences and sort by original position
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
            summary_sentences = [sentences[i] for i, _ in sorted(top_sentences, key=lambda x: x[0])]
            summary = " ".join(summary_sentences)
                          </code>
                        </pre>
                      </div>
                    </div>
                    
                    <div className="method-note">Preserves original wording and focuses on key sentences</div>
                  </div>
                </div>
                
                <h4 className="processing-section-title">Style Parameter Customization</h4>
                <div className="style-parameters-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Style</th>
                        <th>Key Parameters</th>
                        <th>Effect</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Default</td>
                        <td>num_beams=4, length_penalty=1.0</td>
                        <td>Balanced approach to quality and length</td>
                      </tr>
                      <tr>
                        <td>Concise</td>
                        <td>length_penalty=0.6</td>
                        <td>Favors shorter outputs while maintaining quality</td>
                      </tr>
                      <tr>
                        <td>Detailed</td>
                        <td>length_penalty=2.0, early_stopping=False</td>
                        <td>Encourages longer, more comprehensive summaries</td>
                      </tr>
                      <tr>
                        <td>Creative</td>
                        <td>temperature=1.2, top_k=50, top_p=0.9</td>
                        <td>Increases diversity and originality in wording</td>
                      </tr>
                      <tr>
                        <td>Bullets</td>
                        <td>format_bullets=True, prefix formatting</td>
                        <td>Structures output as bullet points for readability</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <h4 className="processing-section-title">Performance Considerations</h4>
                <div className="performance-metrics">
                  <div className="performance-metric">
                    <div className="metric-icon"><i className="fas fa-tachometer-alt"></i></div>
                    <div className="metric-name">Processing Time</div>
                    <div className="metric-value">0.5-3 seconds</div>
                    <div className="metric-note">Varies based on text length and style</div>
                  </div>
                  <div className="performance-metric">
                    <div className="metric-icon"><i className="fas fa-compress-arrows-alt"></i></div>
                    <div className="metric-name">Compression Ratio</div>
                    <div className="metric-value">70-95%</div>
                    <div className="metric-note">Typical text reduction percentage</div>
                  </div>
                  <div className="performance-metric">
                    <div className="metric-icon"><i className="fas fa-microchip"></i></div>
                    <div className="metric-name">Hardware</div>
                    <div className="metric-value">GPU/CPU</div>
                    <div className="metric-note">Adaptive processing based on availability</div>
                  </div>
                  <div className="performance-metric">
                    <div className="metric-icon"><i className="fas fa-network-wired"></i></div>
                    <div className="metric-name">Architecture</div>
                    <div className="metric-value">REST API</div>
                    <div className="metric-note">FastAPI with PyTorch backend</div>
                  </div>
                </div>
              </div>
            </div>
        );
        
      case 'output':
        return (
          <div className="tab-content-pane">
            <h3>Summary Output Features</h3>
            <p>Our application provides rich output options for your summaries:</p>
            
            <div className="output-features">
              <div className="output-feature">
                <div className="feature-icon">
                  <i className="fas fa-file-alt"></i>
                </div>
                <div className="feature-text">
                  <h4>Text Summary</h4>
                  <p>Clean, formatted text summary of your content with compression statistics</p>
                </div>
              </div>
              
              <div className="output-feature">
                <div className="feature-icon">
                  <i className="fas fa-volume-up"></i>
                </div>
                <div className="feature-text">
                  <h4>Text-to-Speech</h4>
                  <p>Listen to your summary with customizable voice options in multiple languages</p>
                </div>
              </div>
              
              <div className="output-feature">
                <div className="feature-icon">
                  <i className="fas fa-chart-pie"></i>
                </div>
                <div className="feature-text">
                  <h4>Summary Statistics</h4>
                  <p>View compression ratio and other metrics about your summary</p>
                </div>
              </div>
              
              <div className="output-feature">
                <div className="feature-icon">
                  <i className="fas fa-user-circle"></i>
                </div>
                <div className="feature-text">
                  <h4>Talking Avatar</h4>
                  <p>Visual representation that reads your summary aloud</p>
                </div>
              </div>
            </div>
          </div>
        );
        
      case 'styles':
        return (
          <div className="tab-content-pane">
            <h3>Summary Style Options</h3>
            <p>Customize your summary with different styles to fit your needs:</p>
            
            <div className="styles-grid">
              <div className="style-example-card">
                <h4>Default</h4>
                <div className="style-badge default">Balanced</div>
                <p>A balanced summary with essential information</p>
              </div>
              
              <div className="style-example-card">
                <h4>Concise</h4>
                <div className="style-badge concise">Brief</div>
                <p>Ultra-short summary with only critical points</p>
              </div>
              
              <div className="style-example-card">
                <h4>Detailed</h4>
                <div className="style-badge detailed">Comprehensive</div>
                <p>Longer summary with more information and context</p>
              </div>
              
              <div className="style-example-card">
                <h4>Bullets</h4>
                <div className="style-badge bullets">Formatted</div>
                <p>• Key points organized as bullet points<br/>• For easy scanning</p>
              </div>
              
              <div className="style-example-card">
                <h4>ELI5</h4>
                <div className="style-badge eli5">Simple</div>
                <p>Explain Like I'm 5 - Uses simple language</p>
              </div>
              
              <div className="style-example-card">
                <h4>Academic</h4>
                <div className="style-badge academic">Formal</div>
                <p>Formal academic style with structured language</p>
              </div>
            </div>
          </div>
        );
        
      default:
        return null;
    }
  };

  return (
    <div className="app-flow-tabs-container">
      <div className="tabs-header">
        {tabs.map(tab => (
          <div 
            key={tab.id}
            className={`tab-item ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <i className={tab.icon}></i>
            <span>{tab.label}</span>
          </div>
        ))}
      </div>
      <div className="tab-content">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default ApplicationFlowTabs;