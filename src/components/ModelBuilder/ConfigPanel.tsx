import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { X } from "lucide-react";

interface ConfigPanelProps {
  selectedNode?: {
    id: string;
    type: string;
    name: string;
    params: Record<string, any>;
  };
  onParamChange?: (id: string, paramName: string, value: any) => void;
  onClose?: () => void;
}

const ConfigPanel = ({
  selectedNode = {
    id: "node-1",
    type: "layer",
    name: "Linear Layer",
    params: {
      in_features: 784,
      out_features: 128,
      bias: true,
      dropout: 0.2,
      activation: "relu",
    },
  },
  onParamChange = () => {},
  onClose = () => {},
}: ConfigPanelProps) => {
  const [activeTab, setActiveTab] = useState("parameters");

  // Helper function to render the appropriate input based on parameter type
  const renderParamInput = (paramName: string, value: any) => {
    if (typeof value === "boolean") {
      return (
        <div className="flex items-center justify-between" key={paramName}>
          <Label htmlFor={paramName} className="flex-1">
            {paramName.replace("_", " ")}
          </Label>
          <Switch
            id={paramName}
            checked={value}
            onCheckedChange={(checked) =>
              onParamChange(selectedNode.id, paramName, checked)
            }
          />
        </div>
      );
    } else if (typeof value === "number") {
      if (value <= 1 && value >= 0) {
        // Likely a probability/percentage value
        return (
          <div className="space-y-2" key={paramName}>
            <div className="flex items-center justify-between">
              <Label htmlFor={paramName}>{paramName.replace("_", " ")}</Label>
              <span className="text-sm text-muted-foreground">
                {value.toFixed(2)}
              </span>
            </div>
            <Slider
              id={paramName}
              min={0}
              max={1}
              step={0.01}
              value={[value]}
              onValueChange={(vals) =>
                onParamChange(selectedNode.id, paramName, vals[0])
              }
            />
          </div>
        );
      } else {
        // Regular number input
        return (
          <div className="space-y-2" key={paramName}>
            <Label htmlFor={paramName}>{paramName.replace("_", " ")}</Label>
            <Input
              id={paramName}
              type="number"
              value={value}
              onChange={(e) =>
                onParamChange(
                  selectedNode.id,
                  paramName,
                  parseFloat(e.target.value),
                )
              }
            />
          </div>
        );
      }
    } else if (paramName === "activation") {
      // Special case for activation functions
      return (
        <div className="space-y-2" key={paramName}>
          <Label htmlFor={paramName}>{paramName.replace("_", " ")}</Label>
          <Select
            value={value}
            onValueChange={(val) =>
              onParamChange(selectedNode.id, paramName, val)
            }
          >
            <SelectTrigger id={paramName}>
              <SelectValue placeholder="Select activation function" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="relu">ReLU</SelectItem>
              <SelectItem value="sigmoid">Sigmoid</SelectItem>
              <SelectItem value="tanh">Tanh</SelectItem>
              <SelectItem value="leaky_relu">Leaky ReLU</SelectItem>
              <SelectItem value="none">None</SelectItem>
            </SelectContent>
          </Select>
        </div>
      );
    } else {
      // Default to text input for strings and other types
      return (
        <div className="space-y-2" key={paramName}>
          <Label htmlFor={paramName}>{paramName.replace("_", " ")}</Label>
          <Input
            id={paramName}
            value={value}
            onChange={(e) =>
              onParamChange(selectedNode.id, paramName, e.target.value)
            }
          />
        </div>
      );
    }
  };

  return (
    <Card className="w-[320px] h-full bg-white dark:bg-gray-800 overflow-auto z-10">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-lg">
          {selectedNode ? selectedNode.name : "Configuration"}
        </CardTitle>
        <button
          onClick={onClose}
          className="rounded-full p-1 hover:bg-gray-100"
          aria-label="Close configuration panel"
        >
          <X className="h-4 w-4" />
        </button>
      </CardHeader>
      <CardContent>
        {selectedNode ? (
          <>
            <Tabs
              defaultValue="parameters"
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            >
              <TabsList className="w-full grid grid-cols-2">
                <TabsTrigger value="parameters">Parameters</TabsTrigger>
                <TabsTrigger value="advanced">Advanced</TabsTrigger>
              </TabsList>
              <TabsContent value="parameters" className="space-y-4 mt-4">
                {Object.entries(selectedNode.params).map(([key, value]) =>
                  renderParamInput(key, value),
                )}
              </TabsContent>
              <TabsContent value="advanced" className="space-y-4 mt-4">
                <div className="space-y-2">
                  <Label htmlFor="node-id">Node ID</Label>
                  <Input
                    id="node-id"
                    value={selectedNode.id}
                    disabled
                    className="bg-gray-50"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="node-type">Node Type</Label>
                  <Input
                    id="node-type"
                    value={selectedNode.type}
                    disabled
                    className="bg-gray-50"
                  />
                </div>
                <div className="p-3 bg-gray-50 rounded-md mt-4">
                  <p className="text-xs text-muted-foreground">
                    Advanced settings allow fine-tuning of model behavior.
                    Changes here may affect model performance and compatibility.
                  </p>
                </div>
              </TabsContent>
            </Tabs>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-[400px] text-center">
            <p className="text-muted-foreground">
              Select a node on the canvas to configure its parameters
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ConfigPanel;
