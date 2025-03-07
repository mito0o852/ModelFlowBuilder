import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import {
  Play,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface TestPanelProps {
  onRunTest?: (inputData: string) => void;
  testResults?: {
    success: boolean;
    message: string;
    details?: string;
  } | null;
  isModelValid?: boolean;
  sidebarOpen?: boolean;
}

const TestPanel = ({
  onRunTest = () => {},
  testResults = null,
  isModelValid = false,
  sidebarOpen = false,
}: TestPanelProps) => {
  const [inputData, setInputData] = useState("[1, 2, 3, 4, 5]");
  const [activeTab, setActiveTab] = useState("input");
  const [isExpanded, setIsExpanded] = useState(false);

  const handleRunTest = () => {
    onRunTest(inputData);
    setActiveTab("results");
    setIsExpanded(true);
  };

  return (
    <div className="absolute bottom-0 left-0 right-0 z-50">
      <Card className="mx-auto max-w-[1000px] bg-white dark:bg-gray-800 border-t shadow-lg">
        <div
          className="absolute -top-8 right-4 bg-white dark:bg-gray-800 rounded-t-lg border border-b-0 px-3 py-1 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {isExpanded ? (
              <>
                <ChevronDown className="h-4 w-4" />
                <span>Hide Test Panel</span>
              </>
            ) : (
              <>
                <ChevronUp className="h-4 w-4" />
                <span>Show Test Panel</span>
              </>
            )}
          </div>
        </div>

        <div
          className={`overflow-hidden transition-all duration-300 ease-in-out ${isExpanded ? "max-h-[400px]" : "max-h-[60px]"}`}
        >
          <CardHeader className="p-3 flex flex-row items-center justify-between border-b">
            <CardTitle className="text-sm font-medium">Test Model</CardTitle>
            <div className="flex items-center gap-2">
              {!isExpanded && testResults && (
                <div className="flex items-center gap-2 text-sm">
                  {testResults.success ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                      <span className="text-green-600">Test Passed</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-4 w-4 text-red-500" />
                      <span className="text-red-600">Test Failed</span>
                    </>
                  )}
                </div>
              )}
              <Button
                size="sm"
                onClick={handleRunTest}
                disabled={!isModelValid}
                className="flex items-center gap-1"
              >
                <Play className="h-3 w-3" />
                Run Test
              </Button>
            </div>
          </CardHeader>

          <CardContent className={`p-0 ${isExpanded ? "block" : "hidden"}`}>
            <Tabs
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            >
              <div className="px-3 pt-2">
                <TabsList className="h-8">
                  <TabsTrigger value="input" className="text-xs">
                    Input Data
                  </TabsTrigger>
                  <TabsTrigger value="results" className="text-xs">
                    Test Results
                  </TabsTrigger>
                  <TabsTrigger value="validation" className="text-xs">
                    Validation
                  </TabsTrigger>
                </TabsList>
              </div>

              <TabsContent value="input" className="px-3 mt-3">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="text-xs text-muted-foreground whitespace-nowrap">
                    Sample Input:
                  </div>
                  <div className="flex-1 min-w-[300px]">
                    <Input
                      value={inputData}
                      onChange={(e) => setInputData(e.target.value)}
                      placeholder="Enter sample input data (e.g., [1, 2, 3, 4, 5])"
                      className="h-8 text-xs"
                    />
                  </div>
                  <div className="text-xs text-muted-foreground whitespace-nowrap">
                    Format: Python list or tensor shape
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="results" className="px-3 mt-3">
                {testResults ? (
                  <div className="flex items-start gap-3 mb-4">
                    {testResults.success ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
                    )}
                    <div>
                      <div className="text-sm font-medium mb-1">
                        {testResults.success ? "Test Passed" : "Test Failed"}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {testResults.message}
                      </div>
                      {testResults.details && (
                        <div className="text-sm mt-2 text-muted-foreground">
                          {testResults.details}
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground py-6 text-center">
                    Run a test to see results
                  </div>
                )}
              </TabsContent>

              <TabsContent value="validation" className="px-3 mt-3 mb-4">
                {isModelValid ? (
                  <Alert
                    variant="default"
                    className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-900"
                  >
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <AlertTitle>Model is valid</AlertTitle>
                    <AlertDescription>
                      All connections and parameters are properly configured.
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Invalid model configuration</AlertTitle>
                    <AlertDescription>
                      Please check for disconnected nodes or incompatible
                      dimensions.
                    </AlertDescription>
                  </Alert>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </div>
      </Card>
    </div>
  );
};

export default TestPanel;
