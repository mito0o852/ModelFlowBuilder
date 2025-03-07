import React, { useState, useCallback, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  NodeTypes,
  EdgeTypes,
  Connection,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
  Handle,
  Position,
  ReactFlow,
  ConnectionLineType,
  reconnectEdge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Button } from "@/components/ui/button";
import { Plus, Minus, Trash2, X } from "lucide-react";

interface ModelNode {
  id: string;
  type: string;
  name: string;
  position: { x: number; y: number };
  inputs: string[];
  outputs: string[];
  config?: Record<string, any>;
}

interface ModelConnection {
  id: string;
  sourceId: string;
  sourceOutput: string;
  targetId: string;
  targetInput: string;
  isValid: boolean;
}

interface CanvasProps {
  nodes?: ModelNode[];
  connections?: ModelConnection[];
  onNodeSelect?: (nodeId: string | null) => void;
  onNodeAdd?: (node: ModelNode) => void;
  onNodeRemove?: (nodeId: string) => void;
  onNodeMove?: (nodeId: string, position: { x: number; y: number }) => void;
  onConnectionCreate?: (
    connection: Omit<ModelConnection, "id" | "isValid">,
  ) => void;
  onConnectionRemove?: (connectionId: string) => void;
  className?: string;
}

// Custom delete confirmation popup
const DeleteConfirmation = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  nodeName 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onConfirm: () => void; 
  nodeName: string;
}) => {
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 max-w-sm w-full">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">Delete Node</h3>
          <button 
            onClick={onClose}
            className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mb-4">Are you sure you want to delete the node "{nodeName}"?</p>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button variant="destructive" onClick={onConfirm}>Delete</Button>
        </div>
      </div>
    </div>
  );
};

// Custom Node Component
const CustomNode = ({
  data,
  selected,
  id,
}: {
  data: any;
  selected: boolean;
  id: string;
}) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  
  const nodeTypeColors = {
    input:
      "bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800 text-blue-900 dark:text-blue-100",
    layer:
      "bg-purple-50 border-purple-200 dark:bg-purple-950 dark:border-purple-800 text-purple-900 dark:text-purple-100",
    activation:
      "bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800 text-green-900 dark:text-green-100",
    operation:
      "bg-orange-50 border-orange-200 dark:bg-orange-950 dark:border-orange-800 text-orange-900 dark:text-orange-100",
    default:
      "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100",
  };

  const nodeColor = nodeTypeColors[data.type] || nodeTypeColors.default;

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setShowDeleteConfirm(true);
  };

  const confirmDelete = () => {
    data.onDelete(id);
    setShowDeleteConfirm(false);
  };

  return (
    <>
      <div
        className={cn(
          "p-4 w-[180px] rounded-md shadow-md transition-all duration-200 border relative",
          nodeColor,
          selected ? "ring-2 ring-primary shadow-lg border-primary" : "",
        )}
      >
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-sm font-medium truncate">{data.name}</h3>
          <button
            className="nodrag absolute top-2 right-2 p-1 rounded-full hover:bg-red-100 dark:hover:bg-red-900/30 text-red-500"
            onClick={handleDelete}
            type="button"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
        <div className="text-xs text-muted-foreground">
          {data.config &&
            Object.entries(data.config).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span>{key}:</span>
                <span className="font-mono">{JSON.stringify(value)}</span>
              </div>
            ))}
        </div>

        {/* Use proper ReactFlow Handle components for connections */}
        {data.inputs && data.inputs.length > 0 && (
          <Handle
            type="target"
            position={Position.Top}
            id="input"
            style={{ background: "#3b82f6", width: "10px", height: "10px" }}
          />
        )}
        {data.outputs && data.outputs.length > 0 && (
          <Handle
            type="source"
            position={Position.Bottom}
            id="output"
            style={{ background: "#3b82f6", width: "10px", height: "10px" }}
          />
        )}
      </div>
      
      <DeleteConfirmation 
        isOpen={showDeleteConfirm} 
        onClose={() => setShowDeleteConfirm(false)} 
        onConfirm={confirmDelete}
        nodeName={data.name}
      />
    </>
  );
};

const nodeTypes: NodeTypes = {
  custom: CustomNode,
};

const Canvas = ({
  nodes = [],
  connections = [],
  onNodeSelect = () => {},
  onNodeAdd = () => {},
  onNodeRemove = () => {},
  onNodeMove = () => {},
  onConnectionCreate = () => {},
  onConnectionRemove = () => {},
  className = "",
}: CanvasProps) => {
  const edgeReconnectSuccessful = useRef(true);
  
  // Convert model nodes to ReactFlow nodes
  const initialNodes: Node[] = nodes.map((node) => ({
    id: node.id,
    type: "custom",
    position: node.position,
    data: {
      name: node.name,
      type: node.type,
      inputs: node.inputs,
      outputs: node.outputs,
      config: node.config,
      onDelete: (nodeId: string) => {
        // Remove all connections associated with this node
        const edgesToRemove = reactFlowEdges.filter(
          (edge) => edge.source === nodeId || edge.target === nodeId
        );
        edgesToRemove.forEach((edge) => {
          onConnectionRemove(edge.id);
        });
        
        // Remove the node from React Flow's state
        setNodes((nodes) => nodes.filter((n) => n.id !== nodeId));
        
        // Remove the node from parent state
        onNodeRemove(nodeId);
      },
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
    connectable: true,
    draggable: true,
  }));

  // Convert model connections to ReactFlow edges
  const initialEdges: Edge[] = connections.map((conn) => ({
    id: conn.id,
    source: conn.sourceId,
    target: conn.targetId,
    type: "smoothstep",
    animated: true,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 20,
      height: 20,
    },
    style: { stroke: "#3b82f6", strokeWidth: 2 },
  }));

  const [reactFlowNodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [reactFlowEdges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Handle node selection
  const onNodeClick = (_: React.MouseEvent, node: Node) => {
    onNodeSelect(node.id);
  };

  // Handle node movement
  const onNodeDragStop = (_: React.MouseEvent, node: Node) => {
    onNodeMove(node.id, node.position);
  };

  // Handle connection creation
  const onConnect = useCallback(
    (params: Connection) => {
      if (params.source && params.target) {
        // Create a new edge in ReactFlow
        setEdges((eds) =>
          addEdge(
            {
              ...params,
              type: "smoothstep",
              animated: true,
              markerEnd: {
                type: MarkerType.ArrowClosed,
                width: 20,
                height: 20,
              },
              style: { stroke: "#3b82f6", strokeWidth: 2 },
            },
            eds,
          ),
        );

        // Notify parent component
        onConnectionCreate({
          sourceId: params.source,
          sourceOutput: params.sourceHandle || "output",
          targetId: params.target,
          targetInput: params.targetHandle || "input",
        });
      }
    },
    [setEdges, onConnectionCreate],
  );

  // Handle edge reconnection
  const onReconnectStart = useCallback(() => {
    edgeReconnectSuccessful.current = false;
  }, []);

  const onReconnect = useCallback((oldEdge: Edge, newConnection: Connection) => {
    edgeReconnectSuccessful.current = true;
    setEdges((els) => reconnectEdge(oldEdge, newConnection, els));
  }, [setEdges]);

  const onReconnectEnd = useCallback((event: MouseEvent | TouchEvent, edge: Edge) => {
    if (!edgeReconnectSuccessful.current) {
      setEdges((eds) => eds.filter((e) => e.id !== edge.id));
      onConnectionRemove(edge.id);
    }
    edgeReconnectSuccessful.current = true;
  }, [setEdges, onConnectionRemove]);

  // Handle edge removal
  const onEdgeClick = (event: React.MouseEvent, edge: Edge) => {
    event.preventDefault();
    event.stopPropagation();
    // Remove the connection without confirmation
    setEdges((eds) => eds.filter((e) => e.id !== edge.id));
    onConnectionRemove(edge.id);
  };

  // Handle node removal
  const onNodesDelete = (nodesToDelete: Node[]) => {
    nodesToDelete.forEach((node) => {
      // Remove all connections associated with this node
      const edgesToRemove = reactFlowEdges.filter(
        (edge) => edge.source === node.id || edge.target === node.id
      );
      edgesToRemove.forEach((edge) => {
        onConnectionRemove(edge.id);
      });
      
      // Remove the node from parent state
      onNodeRemove(node.id);
    });
  };

  // Handle dropping components onto the canvas
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      console.log("Drop event triggered");

      // Get the component data from the drag event
      const componentData = event.dataTransfer.getData("application/json");
      console.log("Component data from drop:", componentData);

      try {
        const component = JSON.parse(componentData);
        console.log("Dropped component:", component);

        // Calculate the drop position relative to the canvas
        const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
        if (!reactFlowBounds) return;

        const position = {
          x: event.clientX - reactFlowBounds.left,
          y: event.clientY - reactFlowBounds.top,
        };
        console.log("Drop position:", position);

        // Generate a unique ID for the new node
        const id = `${component.type}-${Date.now()}`;

        // Create a new node object
        const newNode: ModelNode = {
          id,
          type: component.type,
          name: component.name,
          position,
          inputs: ["input"],
          outputs: ["output"],
          config: component.config || {},
        };
        console.log("Creating new node:", newNode);

        // Add the new node to ReactFlow's state
        const reactFlowNode: Node = {
          id: newNode.id,
          type: "custom",
          position: newNode.position,
          data: {
            name: newNode.name,
            type: newNode.type,
            inputs: newNode.inputs,
            outputs: newNode.outputs,
            config: newNode.config,
            onDelete: (nodeId: string) => {
              // Remove all connections associated with this node
              const edgesToRemove = reactFlowEdges.filter(
                (edge) => edge.source === nodeId || edge.target === nodeId
              );
              edgesToRemove.forEach((edge) => {
                onConnectionRemove(edge.id);
              });
              
              // Remove the node from React Flow's state
              setNodes((nodes) => nodes.filter((n) => n.id !== nodeId));
              
              // Remove the node from parent state
              onNodeRemove(nodeId);
            },
          },
          sourcePosition: Position.Bottom,
          targetPosition: Position.Top,
          connectable: true,
          draggable: true,
        };

        // Update the React Flow nodes state
        setNodes((nds) => nds.concat(reactFlowNode));

        // Notify parent component
        onNodeAdd(newNode);
      } catch (error) {
        console.error("Error adding component:", error);
      }
    },
    [reactFlowEdges, onConnectionRemove, onNodeRemove, onNodeAdd, setNodes],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    console.log("Dragging over canvas");
  }, []);

  return (
    <div
      className={cn("w-full h-full", className)}
      ref={reactFlowWrapper}
      onDrop={onDrop}
      onDragOver={onDragOver}
    >
      <ReactFlow
        nodes={reactFlowNodes}
        edges={reactFlowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onNodeDragStop={onNodeDragStop}
        onEdgeClick={onEdgeClick}
        onNodesDelete={onNodesDelete}
        onReconnectStart={onReconnectStart}
        onReconnect={onReconnect}
        onReconnectEnd={onReconnectEnd}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-right"
        deleteKeyCode={["Backspace", "Delete"]}
        defaultEdgeOptions={{ type: "smoothstep" }}
        connectionLineType={ConnectionLineType.SmoothStep}
        connectionLineStyle={{ stroke: "#3b82f6", strokeWidth: 2 }}
        snapToGrid={true}
        snapGrid={[15, 15]}
        elementsSelectable={true}
        selectNodesOnDrag={false}
        zoomOnScroll={true}
        zoomOnDoubleClick={true}
        panOnScroll={true}
        panOnDrag={true}
      >
        <Background color="#aaa" gap={16} />
        <Controls />
        <MiniMap
          nodeStrokeColor={(n) => {
            if (n.type === "custom") return "#3b82f6";
            return "#eee";
          }}
          nodeColor={(n) => {
            const type = n.data?.type || "default";
            switch (type) {
              case "input":
                return "#93c5fd";
              case "layer":
                return "#c4b5fd";
              case "activation":
                return "#86efac";
              case "operation":
                return "#fdba74";
              default:
                return "#f9fafb";
            }
          }}
          nodeBorderRadius={2}
        />
        <div className="absolute top-4 right-4 z-10">
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => {
                const flow = document.querySelector(".react-flow");
                if (flow) {
                  flow.dispatchEvent(new WheelEvent("wheel", { deltaY: -100 }));
                }
              }}
            >
              <Plus className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={() => {
                const flow = document.querySelector(".react-flow");
                if (flow) {
                  flow.dispatchEvent(new WheelEvent("wheel", { deltaY: 100 }));
                }
              }}
            >
              <Minus className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </ReactFlow>
    </div>
  );
};

export default Canvas;
