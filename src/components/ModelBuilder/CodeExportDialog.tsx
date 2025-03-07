import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Check, Copy, Download, Code, FileEdit } from "lucide-react";

interface CodeExportDialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  generatedCode?: string;
  modelName?: string;
}

const CodeExportDialog = ({
  open = true,
  onOpenChange = () => {},
  generatedCode = `import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create an instance of the model
model = NeuralNetwork()`,
  modelName = "MyNeuralNetwork",
}: CodeExportDialogProps) => {
  const [code, setCode] = useState(generatedCode);
  const [activeTab, setActiveTab] = useState("preview");
  const [copied, setCopied] = useState(false);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadCode = () => {
    const element = document.createElement("a");
    const file = new Blob([code], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `${modelName.replace(/\s+/g, "_").toLowerCase()}.py`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[800px] max-h-[90vh] flex flex-col bg-white">
        <DialogHeader>
          <DialogTitle>Export PyTorch Code</DialogTitle>
          <DialogDescription>
            Review, edit, and export the generated PyTorch code for your neural
            network model.
          </DialogDescription>
        </DialogHeader>

        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          className="flex-1 overflow-hidden flex flex-col"
        >
          <TabsList className="grid grid-cols-2 w-[400px] mx-auto mb-4">
            <TabsTrigger value="preview" className="flex items-center gap-2">
              <Code className="h-4 w-4" />
              Preview Code
            </TabsTrigger>
            <TabsTrigger value="edit" className="flex items-center gap-2">
              <FileEdit className="h-4 w-4" />
              Edit Code
            </TabsTrigger>
          </TabsList>

          <TabsContent
            value="preview"
            className="flex-1 overflow-auto data-[state=active]:flex flex-col"
          >
            <div className="relative rounded-md bg-gray-900 text-gray-100 p-4 overflow-auto h-[400px]">
              <pre className="font-mono text-sm">{code}</pre>
            </div>
          </TabsContent>

          <TabsContent
            value="edit"
            className="flex-1 overflow-auto data-[state=active]:flex flex-col"
          >
            <Textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="font-mono text-sm h-[400px] resize-none bg-gray-50"
              spellCheck="false"
            />
          </TabsContent>
        </Tabs>

        <DialogFooter className="flex flex-row justify-between items-center mt-4 sm:mt-0">
          <div className="text-sm text-muted-foreground">
            Model: <span className="font-medium">{modelName}</span>
          </div>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              onClick={handleCopyCode}
              className="flex items-center gap-2"
            >
              {copied ? (
                <>
                  <Check className="h-4 w-4" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4" />
                  Copy Code
                </>
              )}
            </Button>
            <Button
              onClick={handleDownloadCode}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Download .py
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default CodeExportDialog;
