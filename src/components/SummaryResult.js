import React from 'react';
import TalkingAvatar from './TalkingAvatar';

const SummaryResult = ({ summary, summarizationType }) => {
  // Define different badge colors and icons based on summarization type
  const getBadgeStyle = () => {
    if (summarizationType === 'abstractive') {
      return {
        backgroundColor: '#3f51b5',
        icon: 'fas fa-brain'
      };
    } else {
      return {
        backgroundColor: '#ff9800',
        icon: 'fas fa-scissors'
      };
    }
  };

  const badgeStyle = getBadgeStyle();

  return (
    <div className="summary-container">
      <div className="summary-header">
        <span 
          className="style-badge" 
          style={{ backgroundColor: badgeStyle.backgroundColor }}
        >
          <i className={badgeStyle.icon} style={{ marginRight: '6px' }}></i>
          {summary.style.replace(/_/g, ' ')}
        </span>
        <p className="style-description">{summary.style_description}</p>
      </div>
      
      {/* Add the TalkingAvatar component */}
      <TalkingAvatar text={summary.summary} />
      
      <div className="summary-content">
        <h3>Summary</h3>
        <div className="summary-text">
          {summary.summary}
        </div>
      </div>
      
      <div className="summary-stats">
        <div className="stat">
          <span className="stat-label">Original Length</span>
          <span className="stat-value">{summary.original_length} characters</span>
        </div>
        <div className="stat">
          <span className="stat-label">Summary Length</span>
          <span className="stat-value">{summary.summary_length} characters</span>
        </div>
        <div className="stat">
          <span className="stat-label">Compression Ratio</span>
          <span className="stat-value">
            {(summary.summary_length / summary.original_length * 100).toFixed(1)}%
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Summarization Type</span>
          <span className="stat-value">
            {summarizationType.charAt(0).toUpperCase() + summarizationType.slice(1)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default SummaryResult;