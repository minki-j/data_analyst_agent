import React, { useState, useEffect } from 'react';
import AnalysisHeader from '@/components/AnalysisHeader';
import StreamingMessages from '@/components/StreamingMessages';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Download, FileText } from 'lucide-react';
import { toast } from '@/components/ui/use-toast';
import { Message } from '@/models/Message';

const AnalysisSteps = () => {
  const [user_message, setUserMessage] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [waitingForInput, setWaitingForInput] = useState(false);
  const [analysisCompleted, setAnalysisCompleted] = useState(false);
  const [currentAnalysisStep, setCurrentAnalysisStep] = useState(1);
  const [finalReport, setFinalReport] = useState<string>('');
  const [showReport, setShowReport] = useState(false);

  // Analysis step configurations
  const analysisSteps = [
    {
      step: 1,
      title: "Define Objective",
      description: "Define the objective of the analysis"
    },
    {
      step: 2,
      title: "Data Cleaning",
      description: "Clean the data to make it ready for analysis"
    },
    {
      step: 3,
      title: "Data Exploration",
      description: "Explore the data to understand the properties"
    },
    {
      step: 4,
      title: "Data Analysis",
      description: "Analyze the data to find the properties that meet the objective"
    },
    {
      step: 5,
      title: "Report",
      description: "The analysis is complete, and the report is ready"
    }
  ];

  const connectWebSocket = () => {
    let websocket: WebSocket;
    try {
      const backendUrl = "http://localhost:8000";
      if (!backendUrl) {
        throw new Error("Backend URL is not configured");
      }

      // Create WebSocket URL
      const wsUrl = backendUrl.replace('http', 'ws') + '/agent/';
      console.log("Connecting to WebSocket:", wsUrl);

      const ws = new WebSocket(wsUrl);
      setWebsocket(ws);

      ws.onopen = () => {
        console.log("WebSocket connected");
        setWebsocket(ws);

        // Send initial message
        if (user_message.trim()) {
          setIsLoading(true);
          ws.send(JSON.stringify({ input: user_message }));
          setMessages([]);
        }
      };

      ws.onmessage = (event) => {
        setIsLoading(false);
        const response = JSON.parse(event.data);
        console.log("WebSocket response:", response);

        if (response.error) {
          toast({
            title: "Error",
            description: response.error,
            variant: "destructive",
            duration: 4000,
          });
          return;
        }

        // Add message to the list, handling oneline messages separately
        if (response.oneline_message) {
          setMessages((prev) => [
            ...prev,
            { oneline_message: response.oneline_message }
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            { summary: response.summary || "Processing", content: response.content || "" }
          ]);
        }

        // Update analysis step if provided
        if (response.current_step && response.current_step <= 5) {
          setCurrentAnalysisStep(response.current_step);
        }

        // Handle final report
        if (response.final_report) {
          setFinalReport(response.final_report);
          setShowReport(true);
          setCurrentAnalysisStep(5);
        }

        // Check if input is required
        if (response.requires_input) {
          setWaitingForInput(true);
          setIsLoading(false);
        }

        // Check if analysis is completed
        if (response.completed) {
          setAnalysisCompleted(true);
          setIsLoading(false);
          setWaitingForInput(false);
          if (!showReport && response.final_report) {
            setFinalReport(response.final_report);
            setShowReport(true);
          }
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        toast({
          title: "Connection Error",
          description: "WebSocket connection failed",
          variant: "destructive",
          duration: 4000,
        });
        setIsLoading(false);
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
        setWebsocket(null);
        setIsLoading(false);
      };

      return ws;
    } catch (error) {
      console.error("Failed to connect to WebSocket:", error);
      toast({
        title: "Error",
        description: "Failed to connect to WebSocket",
        variant: "destructive",
        duration: 4000,
      });
      return;
    }
    return websocket;
  };

  const invoke_agent = async (payload: any) => {
    console.log("invoke_agent");

    const websocket = connectWebSocket();
    if (!websocket) {
      return;
    }

    try {
      await Promise.race([
        new Promise((resolve, reject) => {
          websocket.onopen = resolve;
          websocket.onerror = reject;
        }),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("WebSocket Connection timeout")),
            5000
          )
        ),
      ]);

      websocket.onmessage = async (event) => {
        setIsLoading(false);
        const response = JSON.parse(event.data);
        console.log("WebSocket response: ", response);

        if (response.error) {
          toast({
            title: "Error",
            description: response.error,
            variant: "destructive",
            duration: 4000,
          });
          return;
        }

        if (response.oneline_message) {
          setMessages((prev: Message[]) => [
            ...prev,
            { oneline_message: response.oneline_message },
          ]);
        } else if (response.summary && response.content) {
          setMessages((prev: Message[]) => [
            ...prev,
            { summary: response.summary, content: response.content },
          ]);
        }

        // Update analysis step if provided
        if (response.current_step && response.current_step <= 5) {
          setCurrentAnalysisStep(response.current_step);
        }

        // Handle final report
        if (response.final_report) {
          setFinalReport(response.final_report);
          setShowReport(true);
          setCurrentAnalysisStep(5);
        }

        // Check if input is required
        if (response.requires_input) {
          setWaitingForInput(true);
          setIsLoading(false);
        }

        // Check if analysis is completed
        if (response.completed) {
          setAnalysisCompleted(true);
          setIsLoading(false);
          setWaitingForInput(false);
        }
      };

      websocket.send(JSON.stringify(payload));
    } catch (error: any) {
      if (websocket.readyState !== WebSocket.CLOSED) {
        websocket.close();
      }
      toast({
        title: "Error",
        description: `${error.message ? error.message : "Sorry something went wrong"}`,
        variant: "destructive",
        duration: 4000,
      });
    }
  };

  const sendMessage = () => {
    if (!websocket || !user_message.trim()) return;

    setIsLoading(true);
    setWaitingForInput(false);

    websocket.send(JSON.stringify({ input: user_message }));
    setUserMessage("");
  };

  const downloadReport = () => {
    if (!finalReport) return;

    const element = document.createElement('a');
    const file = new Blob([finalReport], { type: 'text/markdown' });
    element.href = URL.createObjectURL(file);
    element.download = `property-analysis-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  useEffect(() => {
    // get form data from local storage
    const analysisFormData = localStorage.getItem('analysisFormData');

    invoke_agent({
      form_data: analysisFormData,
    });
  }, []);

  const currentStep = analysisSteps.find(step => step.step === currentAnalysisStep) || analysisSteps[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <AnalysisHeader currentStep={currentAnalysisStep} totalSteps={5} />

      <div className="max-w-6xl mx-auto px-6 py-8">
        {showReport && finalReport ? (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">Analysis Complete!</h2>
              <p className="text-lg text-gray-600">Your comprehensive property investment report is ready</p>
            </div>

            <Card className="shadow-xl">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <FileText className="h-8 w-8 text-blue-600 mr-3" />
                    <CardTitle className="text-2xl">Investment Analysis Report</CardTitle>
                  </div>
                  <Button onClick={downloadReport} className="bg-green-600 hover:bg-green-700">
                    <Download className="h-4 w-4 mr-2" />
                    Download Report
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[600px] w-full rounded border bg-white p-6">
                  <div className="prose prose-lg max-w-none">
                    <pre className="whitespace-pre-wrap text-sm font-mono text-gray-800 leading-relaxed">
                      {finalReport}
                    </pre>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div>
            <StreamingMessages
              messages={messages}
              isLoading={isLoading}
              stepTitle={currentStep.title}
              stepDescription={currentStep.description}
            />

            {(waitingForInput || analysisCompleted || !isLoading) && (
              <div className="mt-8">
                <div className="flex gap-4">
                  <input
                    type="text"
                    value={user_message}
                    onChange={(e) => setUserMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Type your message here when the agent asks for your input."
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={isLoading || !user_message.trim()}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Send
                  </button>
                </div>
                {waitingForInput && (
                  <div className="mt-4 p-4 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg animate-pulse">
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-amber-400 animate-bounce" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-amber-800">
                          🤖 <strong>Waiting for your response!</strong>
                        </p>
                        <p className="text-sm text-amber-700 mt-1">
                          The AI agent needs your input to continue the analysis. Please type your response above and hit Enter or click Send.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisSteps;
