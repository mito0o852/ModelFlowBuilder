import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Play, AlertCircle, CheckCircle2, X } from "lucide-react";

interface TestPanelProps {
  onRunTest?: (inputData: string) => void;
  testResults?: {
    success: boolean;
    message: string;
    details?: string;
  };
  isModelValid?: boolean;
}

const TestPanel = ({
  onRunTest = () => {},
  testResults = null,
  isModelValid = false,
}: TestPanelProps) => {
  const [inputData, setInputData] = useState("[1, 2, 3, 4, 5]");
  const [activeTab, setActiveTab] = useState("input");

  const handleRunTest = () => {
    onRunTest(inputData);
    setActiveTab("results");
  };

  return (
    <Card className="w-full h-[120px] bg-white border-t z-10">
      <CardHeader className="p-3 flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium">Test Model</CardTitle>
        <Button
          size="sm"
          onClick={handleRunTest}
          disabled={!isModelValid}
          className="flex items-center gap-1"
        >
          <Play className="h-3 w-3" />
          Run Test
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="px-3">
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

          <TabsContent value="input" className="px-3 mt-1">
            <div className="flex items-center gap-2">
              <div className="text-xs text-muted-foreground">Sample Input:</div>
              <Input
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                placeholder="Enter sample input data (e.g., [1, 2, 3, 4, 5])"
                className="h-7 text-xs"
              />
              <div className="text-xs text-muted-foreground">
                Format: Python list or tensor shape
              </div>
            </div>
          </TabsContent>

          <TabsContent value="results" className="px-3 mt-1">
            {testResults ? (
              <div className="flex items-start gap-2">
                {testResults.success ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-red-500 mt-0.5" />
                )}
                <div>
                  <div className="text-sm font-medium">
                    {testResults.success ? "Test Passed" : "Test Failed"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {testResults.message}
                  </div>
                  {testResults.details && (
                    <div className="text-xs mt-1 text-muted-foreground">
                      {testResults.details}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">
                Run a test to see results
              </div>
            )}
          </TabsContent>

          <TabsContent value="validation" className="px-3 mt-1">
            {isModelValid ? (
              <Alert
                variant="default"
                className="py-2 bg-green-50 border-green-200"
              >
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <AlertTitle>Model is valid</AlertTitle>
                <AlertDescription>
                  All connections and parameters are properly configured.
                </AlertDescription>
              </Alert>
            ) : (
              <Alert variant="destructive" className="py-2">
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
    </Card>
  );
};

export default TestPanel;
