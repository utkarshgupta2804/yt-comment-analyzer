import { useState } from "react";
import axios from "axios";
import "./Analyze.css"; // Make sure to create this CSS file

const Analyze = () => {
    const [url, setUrl] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const analyzeVideo = async () => {
        if (!url.trim()) {
            setError("Please enter a YouTube URL");
            return;
        }

        setLoading(true);
        setError(null);
        
        try {
            const response = await axios.post("http://localhost:5000/analyze", { url });
            setResult(response.data);
        } catch (error) {
            console.error("Error:", error);
            setError(error.response?.data?.error || "An error occurred during analysis");
        } finally {
            setLoading(false);
        }
    };

    // Function to determine sentiment color
    const getSentimentColor = (percentage) => {
        if (percentage >= 60) return "#4CAF50"; // Green for high positive
        if (percentage >= 40) return "#8BC34A"; // Light green for medium positive
        if (percentage >= 30) return "#FFC107"; // Amber for neutral-positive
        if (percentage >= 20) return "#FF9800"; // Orange for neutral-negative
        return "#F44336"; // Red for negative
    };

    return (
        <div className="analyzer-container">
            <div className="analyzer-card">
                <div className="analyzer-header">
                    <h1>YouTube Sentiment Analyzer</h1>
                    <p className="analyzer-subtitle">Discover what viewers really think about any YouTube video</p>
                </div>

                <div className="url-input-container">
                    <input
                        type="text"
                        placeholder="Paste YouTube video URL here..."
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        className="url-input"
                    />
                    <button 
                        onClick={analyzeVideo} 
                        className="analyze-button"
                        disabled={loading}
                    >
                        {loading ? "Analyzing..." : "Analyze"}
                    </button>
                </div>

                {error && <div className="error-message">{error}</div>}

                {loading && (
                    <div className="loading-container">
                        <div className="loading-spinner"></div>
                        <p>Analyzing comments... This may take a moment.</p>
                    </div>
                )}

                {result && !loading && (
                    <div className="results-container">
                        <h2>Analysis Results</h2>
                        <div className="video-info">
                            <p><strong>Video ID:</strong> {result.video_id}</p>
                            <p><strong>Comments Analyzed:</strong> {result.total_comments}</p>
                        </div>

                        <h3>Sentiment Breakdown</h3>
                        <div className="sentiment-container">
                            <div className="sentiment-meter">
                                {/* Positive */}
                                <div className="meter-section positive" style={{
                                    width: `${result.sentiment.positive}%`,
                                    backgroundColor: getSentimentColor(result.sentiment.positive)
                                }}>
                                    {result.sentiment.positive > 10 && `${result.sentiment.positive.toFixed(1)}%`}
                                </div>
                                
                                {/* Neutral */}
                                <div className="meter-section neutral" style={{
                                    width: `${result.sentiment.neutral}%`,
                                    backgroundColor: "#607D8B"
                                }}>
                                    {result.sentiment.neutral > 10 && `${result.sentiment.neutral.toFixed(1)}%`}
                                </div>
                                
                                {/* Negative */}
                                <div className="meter-section negative" style={{
                                    width: `${result.sentiment.negative}%`,
                                    backgroundColor: "#F44336"
                                }}>
                                    {result.sentiment.negative > 10 && `${result.sentiment.negative.toFixed(1)}%`}
                                </div>
                            </div>
                            
                            <div className="sentiment-legend">
                                <div className="legend-item">
                                    <span className="legend-color" style={{ backgroundColor: "#4CAF50" }}></span>
                                    <span>Positive: {result.sentiment.positive.toFixed(1)}%</span>
                                </div>
                                <div className="legend-item">
                                    <span className="legend-color" style={{ backgroundColor: "#607D8B" }}></span>
                                    <span>Neutral: {result.sentiment.neutral.toFixed(1)}%</span>
                                </div>
                                <div className="legend-item">
                                    <span className="legend-color" style={{ backgroundColor: "#F44336" }}></span>
                                    <span>Negative: {result.sentiment.negative.toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>

                        {/* Additional content if available */}
                        {result.channel_info && (
                            <div className="channel-info">
                                <h3>Channel Information</h3>
                                <p><strong>Channel:</strong> {result.channel_info.channel_title}</p>
                                <p><strong>Subscribers:</strong> {parseInt(result.channel_info.subscriber_count).toLocaleString()}</p>
                            </div>
                        )}

                        {result.content_ideas && result.content_ideas.length > 0 && (
                            <div className="content-ideas">
                                <h3>Viewer Interests</h3>
                                <ul>
                                    {result.content_ideas.map((idea, index) => (
                                        <li key={index}>{idea}</li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {result.common_questions && result.common_questions.length > 0 && (
                            <div className="common-questions">
                                <h3>Common Questions</h3>
                                <ul>
                                    {result.common_questions.map((question, index) => (
                                        <li key={index}>{question}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Analyze;