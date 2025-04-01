
"use client";

import React, { useState, useEffect, useRef } from 'react';

// Message type definition
interface Message {
    id: string;
    text: string;
    sender: 'user' | 'system';
    timestamp: Date;
}

const Chat: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto scroll to bottom when messages update
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSendMessage = async () => {
        if (!inputText.trim() || isLoading) return;
        
        // Create new user message
        const userMessage: Message = {
            id: Date.now().toString(),
            text: inputText,
            sender: 'user',
            timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, userMessage]);
        const userComment = inputText;
        setInputText('');
        setIsLoading(true);
        
        try {
            // Send request to the prediction API
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ comment: userComment }),
            });
            
            if (!response.ok) {
                throw new Error('Failed to get response');
            }
            
            const data = await response.json();
            console.log('Prediction:', data);
            
            // Add response to messages
            const systemMessage: Message = {
                id: Date.now().toString(),
                text: typeof data === 'object' ? JSON.stringify(data) : String(data),
                sender: 'system',
                timestamp: new Date(),
            };
            
            setMessages(prev => [...prev, systemMessage]);
        } catch (error) {
            console.error('Error:', error);
            
            // Add error message
            const errorMessage: Message = {
                id: Date.now().toString(),
                text: 'Failed to send message. Please try again.',
                sender: 'system',
                timestamp: new Date(),
            };
            
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen w-full border rounded-lg shadow-md px-100">
            {/* Chat messages */}
            <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
                {messages.map((message) => (
                    <div 
                        key={message.id} 
                        className={`mb-4 ${message.sender === 'user' ? 'text-right' : 'text-left'}`}
                    >
                        <div 
                            className={`inline-block p-3 rounded-lg ${
                                message.sender === 'user' 
                                    ? 'bg-blue-500 text-white' 
                                    : 'bg-gray-200 text-gray-800'
                            }`}
                        >
                            {message.text}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                            {message.timestamp.toLocaleTimeString()}
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            
            {/* Input area */}
            <div className="border-t p-3 flex bg-white">
                <input
                    type="text"
                    className="flex-1 border rounded-l-lg px-3 py-2 focus:outline-none"
                    placeholder="Type your message..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    disabled={isLoading}
                />
                <button
                    className={`px-4 py-2 rounded-r-lg focus:outline-none ${
                        isLoading 
                            ? 'bg-gray-400 text-white' 
                            : 'bg-blue-500 text-white hover:bg-blue-600'
                    }`}
                    onClick={handleSendMessage}
                    disabled={isLoading}
                >
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
};

export default Chat;