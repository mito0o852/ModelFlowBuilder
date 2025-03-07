import React, { useState } from "react";
import Header from "./ModelBuilder/Header";
import Canvas from "./ModelBuilder/Canvas";
import ComponentLibrary from "./ModelBuilder/ComponentLibrary";
import ConfigPanel from "./ModelBuilder/ConfigPanel";
import TestPanel from "./ModelBuilder/TestPanel";
import CodeExportDialog from "./ModelBuilder/CodeExportDialog";
import SaveLoadDialog from "./ModelBuilder/SaveLoadDialog";

interface Node {
  id: string;
  type: string;
  name: string;
  position: { x: number; y: number };
  inputs: string[];
  outputs: string[];
  config?: Record<string, any>;
}

interface Connection {
  id: string;
  sourceId: string;
  sourceOutput: string;
  targetId: string;
  targetInput: string;
  isValid: boolean;
}

const Home = () => {
  // State for nodes and connections
  const [nodes, setNodes] = useState<Node[]>([
    {
      id: "input-1",
      type: "input",
      name: "Input Layer",
      position: { x: 100, y: 150 },
      inputs: [],
      outputs: ["output"],
      config: { dimensions: [1, 28, 28] },
    },
    {
      id: "conv-1",
      type: "layer",
      name: "Conv2d",
      position: { x: 350, y: 150 },
      inputs: ["input"],
      outputs: ["output"],
      config: { in_channels: 1, out_channels: 32, kernel_size: 3 },
    },
    {
      id: "relu-1",
      type: "activation",
      name: "ReLU",
      position: { x: 600, y: 150 },
      inputs: ["input"],
      outputs: ["output"],
      config: {},
    },
  ]);

  const [connections, setConnections] = useState<Connection[]>([
    {
      id: "conn-1",
      sourceId: "input-1",
      sourceOutput: "output",
      targetId: "conv-1",
      targetInput: "input",
      isValid: true,
    },
    {
      id: "conn-2",
      sourceId: "conv-1",
      sourceOutput: "output",
      targetId: "relu-1",
      targetInput: "input",
      isValid: true,
    },
  ]);

  // State for selected node
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const selectedNode = selectedNodeId
    ? nodes.find((node) => node.id === selectedNodeId)
    : null;

  // State for dialogs
  const [codeExportOpen, setCodeExportOpen] = useState(false);
  const [saveLoadOpen, setSaveLoadOpen] = useState(false);

  // State for model validation and testing
  const [isModelValid, setIsModelValid] = useState(true);
  const [testResults, setTestResults] = useState<{
    success: boolean;
    message: string;
    details?: string;
  } | null>(null);

  // State for model name
  const [modelName, setModelName] = useState("MyNeuralNetwork");

  // Handler for node selection
  const handleNodeSelect = (nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  };

  // Handler for node addition
  const handleNodeAdd = (node: Node) => {
    setNodes((prev) => [...prev, node]);
  };

  // Handler for node removal
  const handleNodeRemove = (nodeId: string) => {
    // Remove connections associated with this node
    setConnections((prev) =>
      prev.filter(
        (conn) => conn.sourceId !== nodeId && conn.targetId !== nodeId,
      ),
    );
    // Remove the node
    setNodes((prev) => prev.filter((node) => node.id !== nodeId));
    // Deselect if this was the selected node
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
    }
  };

  // Handler for node movement
  const handleNodeMove = (
    nodeId: string,
    position: { x: number; y: number },
  ) => {
    setNodes((prev) =>
      prev.map((node) => (node.id === nodeId ? { ...node, position } : node)),
    );
  };

  // Handler for connection creation
  const handleConnectionCreate = (
    connection: Omit<Connection, "id" | "isValid">,
  ) => {
    // Check if connection already exists
    const connectionExists = connections.some(
      (conn) =>
        conn.sourceId === connection.sourceId &&
        conn.sourceOutput === connection.sourceOutput &&
        conn.targetId === connection.targetId &&
        conn.targetInput === connection.targetInput,
    );

    if (!connectionExists) {
      // In a real app, we would validate the connection here
      const isValid = true;
      setConnections((prev) => [
        ...prev,
        {
          id: `conn-${Date.now()}`,
          ...connection,
          isValid,
        },
      ]);
    }
  };

  // Handler for connection removal
  const handleConnectionRemove = (connectionId: string) => {
    setConnections((prev) => prev.filter((conn) => conn.id !== connectionId));
  };

  // Handler for parameter changes
  const handleParamChange = (nodeId: string, paramName: string, value: any) => {
    setNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              config: { ...node.config, [paramName]: value },
            }
          : node,
      ),
    );
  };

  // Handler for running tests
  const handleRunTest = (inputData: string) => {
    // In a real app, we would validate the model and run the test
    // For now, we'll just simulate a successful test
    setTestResults({
      success: true,
      message: "Model successfully processed the input data",
      details: `Input shape: [1, 28, 28], Output shape: [1, 10]`,
    });
  };

  // Generate PyTorch code based on the model
  const generatePyTorchCode = () => {
    // In a real app, this would generate actual code based on the model structure
    return `import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model = NeuralNetwork()`;
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col">
      <Header
        onSave={() => setSaveLoadOpen(true)}
        onLoad={() => setSaveLoadOpen(true)}
        onExport={() => setCodeExportOpen(true)}
        onTest={() => handleRunTest("[1, 28, 28]")}
      />
      <div className="flex-1 flex overflow-hidden neural-network-builder">
        <div className="border-r border-border overflow-hidden">
          <ComponentLibrary
            onComponentSelect={(component) => {
              // In a real app, this would create a new node based on the component
              console.log("Selected component:", component);
            }}
          />
        </div>
        <div className="flex-1 flex">
          <div className="flex-1 relative overflow-hidden">
            <Canvas
              nodes={nodes}
              connections={connections}
              onNodeSelect={handleNodeSelect}
              onNodeAdd={handleNodeAdd}
              onNodeRemove={handleNodeRemove}
              onNodeMove={handleNodeMove}
              onConnectionCreate={handleConnectionCreate}
              onConnectionRemove={handleConnectionRemove}
              className="scrollable-area"
            />
          </div>
          {selectedNode && (
            <div className="w-[300px] border-l border-border overflow-hidden">
              <ConfigPanel
                selectedNode={{
                  id: selectedNode.id,
                  type: selectedNode.type,
                  name: selectedNode.name,
                  params: selectedNode.config || {}
                }}
                onParamChange={handleParamChange}
                onClose={() => setSelectedNodeId(null)}
              />
            </div>
          )}
        </div>
      </div>

      {/* Test Panel */}
      <TestPanel
        onRunTest={handleRunTest}
        testResults={testResults}
        isModelValid={isModelValid}
        sidebarOpen={!!selectedNode}
      />

      {/* Dialogs */}
      {codeExportOpen && (
        <CodeExportDialog
          open={codeExportOpen}
          onOpenChange={setCodeExportOpen}
          generatedCode={generatePyTorchCode()}
          modelName="MyNeuralNetwork"
        />
      )}

      <SaveLoadDialog
        open={saveLoadOpen}
        onOpenChange={setSaveLoadOpen}
        onSave={(name, description) => {
          console.log("Saving model:", name, description);
          setSaveLoadOpen(false);
        }}
        onLoad={(modelId) => {
          console.log("Loading model:", modelId);
          setSaveLoadOpen(false);
        }}
        onDelete={(modelId) => {
          console.log("Deleting model:", modelId);
        }}
      />
    </div>
  );
};

export default Home;
