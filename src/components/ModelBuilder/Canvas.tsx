import React, { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Plus, Minus, Move, Trash2, Link, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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

interface CanvasProps {
  nodes?: Node[];
  connections?: Connection[];
  onNodeSelect?: (nodeId: string | null) => void;
  onNodeAdd?: (node: Node) => void;
  onNodeRemove?: (nodeId: string) => void;
  onNodeMove?: (nodeId: string, position: { x: number; y: number }) => void;
  onConnectionCreate?: (connection: Omit<Connection, "id" | "isValid">) => void;
  onConnectionRemove?: (connectionId: string) => void;
  className?: string;
}

const Canvas = ({
  nodes = [
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
  ],
  connections = [
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
  ],
  onNodeSelect = () => {},
  onNodeAdd = () => {},
  onNodeRemove = () => {},
  onNodeMove = () => {},
  onConnectionCreate = () => {},
  onConnectionRemove = () => {},
  className = "",
}: CanvasProps) => {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [creatingConnection, setCreatingConnection] = useState<{
    sourceId: string;
    sourceOutput: string;
    position: { x: number; y: number };
  } | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);

  // Handle node selection
  const handleNodeSelect = (nodeId: string) => {
    setSelectedNode(nodeId);
    onNodeSelect(nodeId);
  };

  // Handle canvas drag events
  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;
    setMousePosition({ x, y });

    // Handle node dragging
    if (draggingNode) {
      const node = nodes.find((n) => n.id === draggingNode);
      if (node) {
        const newPosition = {
          x: x - dragOffset.x,
          y: y - dragOffset.y,
        };
        onNodeMove(draggingNode, newPosition);
      }
    }
  };

  // Handle node drag start
  const handleNodeDragStart = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    setDraggingNode(nodeId);

    const node = nodes.find((n) => n.id === nodeId);
    if (node && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) / zoom;
      const y = (e.clientY - rect.top) / zoom;
      setDragOffset({
        x: x - node.position.x,
        y: y - node.position.y,
      });
    }
  };

  // Handle node drag end
  const handleNodeDragEnd = () => {
    setDraggingNode(null);
  };

  // Handle canvas drop for new components
  const handleCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (!canvasRef.current) return;

    try {
      const data = JSON.parse(e.dataTransfer.getData("application/json"));
      const rect = canvasRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) / zoom;
      const y = (e.clientY - rect.top) / zoom;

      const newNode: Node = {
        id: `${data.type}-${Date.now()}`,
        type: data.type,
        name: data.name,
        position: { x, y },
        inputs: ["input"],
        outputs: ["output"],
        config: {},
      };

      onNodeAdd(newNode);
      handleNodeSelect(newNode.id);
    } catch (error) {
      console.error("Error adding new node:", error);
    }
  };

  // Handle connection creation start
  const handleConnectionStart = (
    nodeId: string,
    outputName: string,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
    if (!canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    setCreatingConnection({
      sourceId: nodeId,
      sourceOutput: outputName,
      position: { x, y },
    });
  };

  // Handle connection creation end
  const handleConnectionEnd = (nodeId: string, inputName: string) => {
    if (creatingConnection && creatingConnection.sourceId !== nodeId) {
      onConnectionCreate({
        sourceId: creatingConnection.sourceId,
        sourceOutput: creatingConnection.sourceOutput,
        targetId: nodeId,
        targetInput: inputName,
      });
    }
    setCreatingConnection(null);
  };

  // Handle connection removal
  const handleConnectionRemove = (
    connectionId: string,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
    onConnectionRemove(connectionId);
  };

  // Handle zoom
  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.1, 2));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 0.1, 0.5));
  };

  // Handle canvas drag over for drop operations
  const handleCanvasDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  // Calculate connection path between nodes
  const getConnectionPath = (connection: Connection) => {
    const sourceNode = nodes.find((n) => n.id === connection.sourceId);
    const targetNode = nodes.find((n) => n.id === connection.targetId);

    if (!sourceNode || !targetNode) return "";

    // Calculate output port position
    const sourceOutputIndex = sourceNode.outputs.indexOf(
      connection.sourceOutput,
    );
    const sourceOutputCount = sourceNode.outputs.length;
    const sourceX = sourceNode.position.x + 150; // Node width
    const sourceY =
      sourceNode.position.y + 40 + (sourceOutputIndex / sourceOutputCount) * 40;

    // Calculate input port position
    const targetInputIndex = targetNode.inputs.indexOf(connection.targetInput);
    const targetInputCount = targetNode.inputs.length;
    const targetX = targetNode.position.x;
    const targetY =
      targetNode.position.y + 40 + (targetInputIndex / targetInputCount) * 40;

    // Create bezier curve
    const controlPointOffset = 50;
    return `M ${sourceX} ${sourceY} C ${sourceX + controlPointOffset} ${sourceY}, ${targetX - controlPointOffset} ${targetY}, ${targetX} ${targetY}`;
  };

  // Calculate temporary connection path during creation
  const getTempConnectionPath = () => {
    if (!creatingConnection) return "";

    const sourceNode = nodes.find((n) => n.id === creatingConnection.sourceId);
    if (!sourceNode) return "";

    // Calculate output port position
    const sourceOutputIndex = sourceNode.outputs.indexOf(
      creatingConnection.sourceOutput,
    );
    const sourceOutputCount = sourceNode.outputs.length;
    const sourceX = sourceNode.position.x + 150; // Node width
    const sourceY =
      sourceNode.position.y + 40 + (sourceOutputIndex / sourceOutputCount) * 40;

    // Create bezier curve to mouse position
    const controlPointOffset = 50;
    return `M ${sourceX} ${sourceY} C ${sourceX + controlPointOffset} ${sourceY}, ${mousePosition.x - controlPointOffset} ${mousePosition.y}, ${mousePosition.x} ${mousePosition.y}`;
  };

  // Handle node removal
  const handleNodeRemove = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedNode === nodeId) {
      setSelectedNode(null);
      onNodeSelect(null);
    }
    onNodeRemove(nodeId);
  };

  // Clean up event listeners
  useEffect(() => {
    const handleMouseUp = () => {
      setDraggingNode(null);
      if (creatingConnection) {
        setCreatingConnection(null);
      }
    };

    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [creatingConnection]);

  return (
    <div
      className={cn(
        "relative w-full h-full overflow-auto scrollable-area",
        className,
      )}
    >
      {/* Zoom controls */}
      <div className="absolute top-4 right-4 z-20 flex gap-2">
        <Button variant="outline" size="icon" onClick={handleZoomIn}>
          <Plus className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="icon" onClick={handleZoomOut}>
          <Minus className="h-4 w-4" />
        </Button>
      </div>

      {/* Canvas */}
      <div
        ref={canvasRef}
        className="w-[2000px] h-[2000px] relative"
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={() => setDraggingNode(null)}
        onDragOver={handleCanvasDragOver}
        onDrop={handleCanvasDrop}
      >
        <div
          className="absolute top-0 left-0 w-full h-full transform-origin-center transition-transform duration-200"
          style={{ transform: `scale(${zoom})` }}
        >
          {/* Grid background */}
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAwIDEwIEwgNDAgMTAgTSAxMCAwIEwgMTAgNDAgTSAwIDIwIEwgNDAgMjAgTSAyMCAwIEwgMjAgNDAgTSAwIDMwIEwgNDAgMzAgTSAzMCAwIEwgMzAgNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2UyZThlYiIgb3BhY2l0eT0iMC4yIiBzdHJva2Utd2lkdGg9IjEiLz48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZTJlOGViIiBzdHJva2Utd2lkdGg9IjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')]" />

          {/* Connections */}
          <svg className="absolute inset-0 pointer-events-none z-0">
            {connections.map((connection) => (
              <g key={connection.id}>
                <path
                  d={getConnectionPath(connection)}
                  stroke={connection.isValid ? "#3b82f6" : "#ef4444"}
                  strokeWidth="2"
                  fill="none"
                  className="pointer-events-auto"
                  onClick={(e) => handleConnectionRemove(connection.id, e)}
                />
                {!connection.isValid && (
                  <g
                    transform={`translate(${(nodes.find((n) => n.id === connection.sourceId)?.position.x || 0) + 75}, ${(nodes.find((n) => n.id === connection.sourceId)?.position.y || 0) + 20})`}
                  >
                    <circle cx="0" cy="0" r="8" fill="#fee2e2" />
                    <AlertCircle
                      className="h-4 w-4 text-red-500"
                      style={{ transform: "translate(-8px, -8px)" }}
                    />
                  </g>
                )}
              </g>
            ))}

            {/* Temporary connection while creating */}
            {creatingConnection && (
              <path
                d={getTempConnectionPath()}
                stroke="#3b82f6"
                strokeWidth="2"
                strokeDasharray="5,5"
                fill="none"
              />
            )}
          </svg>

          {/* Nodes */}
          {nodes.map((node) => (
            <div
              key={node.id}
              className={cn(
                "absolute p-4 w-[150px] rounded-md shadow-md transition-shadow",
                selectedNode === node.id
                  ? "ring-2 ring-primary shadow-lg"
                  : "shadow",
                node.type === "input"
                  ? "bg-blue-50"
                  : node.type === "layer"
                    ? "bg-purple-50"
                    : node.type === "activation"
                      ? "bg-green-50"
                      : node.type === "operation"
                        ? "bg-orange-50"
                        : "bg-white",
              )}
              style={{
                left: `${node.position.x}px`,
                top: `${node.position.y}px`,
                cursor: draggingNode === node.id ? "grabbing" : "grab",
                zIndex: selectedNode === node.id ? 5 : 1,
              }}
              onClick={() => handleNodeSelect(node.id)}
              onMouseDown={(e) => handleNodeDragStart(e, node.id)}
              onMouseUp={handleNodeDragEnd}
            >
              {/* Node header */}
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-sm font-medium truncate">{node.name}</h3>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        className="text-gray-500 hover:text-red-500 transition-colors"
                        onClick={(e) => handleNodeRemove(node.id, e)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Remove node</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>

              {/* Node inputs */}
              <div className="space-y-1">
                {node.inputs.map((input, index) => (
                  <div
                    key={`${node.id}-input-${index}`}
                    className="flex items-center"
                  >
                    <div
                      className="w-3 h-3 rounded-full bg-blue-500 cursor-pointer mr-2 relative -ml-5"
                      onClick={() =>
                        creatingConnection &&
                        handleConnectionEnd(node.id, input)
                      }
                    />
                    <span className="text-xs text-gray-600">{input}</span>
                  </div>
                ))}
              </div>

              {/* Node outputs */}
              <div className="space-y-1 mt-2">
                {node.outputs.map((output, index) => (
                  <div
                    key={`${node.id}-output-${index}`}
                    className="flex items-center justify-end"
                  >
                    <span className="text-xs text-gray-600">{output}</span>
                    <div
                      className="w-3 h-3 rounded-full bg-blue-500 cursor-pointer ml-2 relative -mr-5"
                      onMouseDown={(e) =>
                        handleConnectionStart(node.id, output, e)
                      }
                    />
                  </div>
                ))}
              </div>

              {/* Node configuration preview */}
              {node.config && Object.keys(node.config).length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
                  {Object.entries(node.config)
                    .slice(0, 2)
                    .map(([key, value]) => (
                      <div key={key} className="truncate">
                        <span className="font-medium">{key}:</span>{" "}
                        {JSON.stringify(value)}
                      </div>
                    ))}
                  {Object.keys(node.config).length > 2 && (
                    <div className="text-xs text-gray-400">
                      + {Object.keys(node.config).length - 2} more...
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Canvas;
