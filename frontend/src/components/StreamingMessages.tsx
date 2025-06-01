import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import { Message } from '@/models/Message';

interface StreamingMessagesProps {
  messages: Message[];
  isLoading: boolean;
  stepTitle: string;
  stepDescription: string;
}

const StreamingMessages = ({ messages, isLoading, stepTitle, stepDescription }: StreamingMessagesProps) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [expandedMessages, setExpandedMessages] = useState<Set<number>>(new Set());

  // Automatically expand all messages by default whenever new messages arrive.
  useEffect(() => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      messages.forEach((_, idx) => newSet.add(idx));
      return newSet;
    });
  }, [messages]);

  // Automatically scroll to the bottom whenever the `messages` array changes.
  // Radix UI's `ScrollArea` puts the scrollable region inside an element with the
  // `data-radix-scroll-area-viewport` attribute, so we target that if it exists.
  useEffect(() => {
    const root = scrollAreaRef.current;
    if (!root) return;

    // Try to find the Radix ScrollArea viewport; fall back to the root itself.
    const viewport = root.querySelector(
      '[data-radix-scroll-area-viewport]'
    ) as HTMLDivElement | null;

    const scrollContainer = viewport ?? root;
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
  }, [messages]);

  const toggleMessage = (index: number) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">{stepTitle}</h2>
        <p className="text-lg text-gray-600">{stepDescription}</p>
      </div>

      <Card className="shadow-lg">
        <CardContent className="p-6">
          <div className="flex items-center mb-4">
            {isLoading && <Loader2 className="h-5 w-5 animate-spin text-blue-600 mr-2" />}
          </div>

          <ScrollArea className="h-96 w-full rounded border bg-gray-50 p-4" ref={scrollAreaRef}>
            <div className="space-y-3">
              {messages.length === 0 && !isLoading && (
                <p className="text-gray-500 italic">Waiting for analysis to begin...</p>
              )}

              {messages.map((message, index) => {
                const isExpanded = expandedMessages.has(index);

                // If oneline_message exists, show simple paragraph (no card/toggle)
                if (message.oneline_message) {
                  return (
                    <p key={index} className="text-sm text-gray-700">
                      {message.oneline_message}
                    </p>
                  );
                }

                // Otherwise show expandable message
                return (
                  <div key={index} className="bg-white rounded-lg shadow-sm border-l-4 border-blue-500">
                    <div
                      className="p-3 cursor-pointer hover:bg-gray-50 transition-colors duration-200"
                      onClick={() => toggleMessage(index)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 mr-2">
                          <div className="text-sm font-medium text-gray-800 mb-1">
                            {message.summary}
                          </div>
                          {isExpanded && (
                            <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono mt-3 pt-3 border-t border-gray-200">
                              {message.content}
                            </pre>
                          )}
                        </div>
                        <div className="flex-shrink-0 ml-2">
                          {isExpanded ? (
                            <ChevronDown className="h-4 w-4 text-gray-500" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-gray-500" />
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="bg-blue-50 rounded-lg p-3 border-l-4 border-blue-400">
                  <div className="flex items-center text-blue-700">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    <span className="text-sm">Processing your request...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
};

export default StreamingMessages;
