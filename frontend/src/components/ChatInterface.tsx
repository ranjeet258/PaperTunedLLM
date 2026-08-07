"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, FileText, Upload, Trash2, Cpu, Settings, Search, Loader2 } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface Document {
  id: string;
  title: string;
  chunks: number;
}

export default function DeepResearchPlatform() {
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'assistant', content: 'Welcome to PaperTunedLLM. Upload a research paper or ask a question to begin your deep research.' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:8080/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content }),
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), role: 'assistant', content: '' }]);

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          
          setMessages(prev => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1].content += chunk;
            return newMessages;
          });
        }
      }
    } catch (error) {
      console.error('Error fetching chat response:', error);
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'assistant', content: 'Sorry, I encountered an error. Please ensure the backend is running.' }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8080/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.status === 'success') {
        setDocuments(prev => [...prev, { id: data.document_id, title: file.name, chunks: data.chunks_indexed }]);
      }
    } catch (error) {
      console.error('Error uploading document:', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0f1115] text-gray-200 font-sans">
      {/* Sidebar - Knowledge Base */}
      <aside className="w-72 bg-[#161b22] border-r border-gray-800 flex flex-col">
        <div className="p-5 border-b border-gray-800">
          <div className="flex items-center gap-2 mb-1">
            <Cpu className="text-blue-500" size={24} />
            <h1 className="text-lg font-bold text-white tracking-wide">PaperTunedLLM</h1>
          </div>
          <p className="text-xs text-gray-500">Deep Research Assistant</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Knowledge Base</h2>
            <label className="cursor-pointer text-blue-400 hover:text-blue-300 transition">
              <Upload size={16} />
              <input type="file" accept=".pdf" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>

          {isUploading && (
            <div className="flex items-center gap-2 text-sm text-gray-400 mb-4 p-2 bg-gray-800/50 rounded-lg">
              <Loader2 size={16} className="animate-spin text-blue-500" />
              Processing document...
            </div>
          )}

          <div className="space-y-2">
            {documents.length === 0 && !isUploading ? (
              <div className="text-sm text-gray-500 text-center p-4 border border-dashed border-gray-700 rounded-lg">
                No papers indexed. Upload a PDF to begin RAG.
              </div>
            ) : (
              documents.map(doc => (
                <div key={doc.id} className="flex items-start justify-between p-3 bg-gray-800/40 hover:bg-gray-800 rounded-lg group transition border border-gray-800/50">
                  <div className="flex gap-3 overflow-hidden">
                    <FileText className="text-blue-400 mt-0.5 shrink-0" size={16} />
                    <div>
                      <p className="text-sm text-gray-300 truncate font-medium">{doc.title}</p>
                      <p className="text-xs text-gray-500 mt-1">{doc.chunks} chunks embedded</p>
                    </div>
                  </div>
                  <button className="text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition shrink-0 p-1">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="p-4 border-t border-gray-800 text-xs text-gray-500 flex items-center gap-2 hover:text-gray-300 cursor-pointer transition">
          <Settings size={14} /> Model Settings (AWQ 4-bit)
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative">
        {/* Header */}
        <header className="h-16 border-b border-gray-800 flex items-center px-6 justify-between bg-[#161b22]/50 backdrop-blur-sm z-10">
          <div className="flex items-center gap-4">
            <span className="px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs font-medium border border-blue-500/20">vLLM Inference</span>
            <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-medium border border-emerald-500/20">Qdrant Active</span>
          </div>
        </header>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map(msg => (
              <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-blue-900/20 border border-blue-400/20">
                    <Cpu size={16} className="text-white" />
                  </div>
                )}
                <div className={`max-w-[80%] rounded-2xl p-5 ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white shadow-xl shadow-blue-900/20 rounded-tr-sm' 
                    : 'bg-[#1e232b] text-gray-300 border border-gray-800/80 rounded-tl-sm shadow-xl'
                }`}>
                  <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.content}</p>
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-gray-700 flex items-center justify-center shrink-0 border border-gray-600">
                    <span className="text-xs font-bold">U</span>
                  </div>
                )}
              </div>
            ))}
            {isTyping && (
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shrink-0 border border-blue-400/20">
                  <Cpu size={16} className="text-white" />
                </div>
                <div className="bg-[#1e232b] border border-gray-800/80 rounded-2xl rounded-tl-sm p-5 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"></span>
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 bg-gradient-to-t from-[#0f1115] via-[#0f1115] to-transparent">
          <div className="max-w-4xl mx-auto relative">
            <div className="relative group flex items-end bg-[#161b22] border border-gray-700 rounded-2xl shadow-2xl focus-within:border-blue-500/50 focus-within:ring-1 focus-within:ring-blue-500/50 transition-all">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask a question based on your uploaded research papers..."
                className="w-full bg-transparent text-gray-200 p-4 min-h-[56px] max-h-48 resize-none focus:outline-none placeholder-gray-500 text-[15px]"
                rows={1}
              />
              <div className="p-2 shrink-0">
                <button 
                  onClick={handleSend}
                  disabled={!input.trim() || isTyping}
                  className="p-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-xl transition-all shadow-lg flex items-center justify-center"
                >
                  <Send size={18} />
                </button>
              </div>
            </div>
            <div className="text-center mt-3 text-xs text-gray-600 font-medium tracking-wide">
              PaperTunedLLM operates on indexed RAG knowledge. Hallucinations are minimized but verify citations.
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
