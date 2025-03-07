import React, { useState, useCallback, useRef } from "react";
import { cn } from "@/lib/utils";
import ReactFlow, {
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
} from "react-flow-renderer";
import { Button } from "@/components/ui/button";
import { Plus, Minus, Trash2 } from "lucide-react";

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
    if (window.confirm("Are you sure you want to delete this node?")) {
      if (data.onDelete) {
        data.onDelete(id);
      }
    }
  };

  return (
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
      onDelete: onNodeRemove,
    },
    sourcePosition: "bottom",
    targetPosition: "top",
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

  // Handle edge removal
  const onEdgeClick = (event: React.MouseEvent, edge: Edge) => {
    event.preventDefault();
    event.stopPropagation();
    if (window.confirm("Are you sure you want to delete this connection?")) {
      setEdges((eds) => eds.filter((e) => e.id !== edge.id));
      onConnectionRemove(edge.id);
    }
  };

  // Handle node removal
  const onNodesDelete = (nodesToDelete: Node[]) => {
    nodesToDelete.forEach((node) => onNodeRemove(node.id));
  };

  // Handle dropping components onto the canvas
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      console.log("Drop event triggered");

      // Get the component data from the drag event
      const componentData = event.dataTransfer.getData("application/json");
      console.log("Component data from drop:", componentData);

      if (componentData) {
        try {
          const component = JSON.parse(componentData);
          console.log("Dropped component:", component);

          // Get the position where the component was dropped
          const reactFlowBounds =
            reactFlowWrapper.current?.getBoundingClientRect();
          if (reactFlowBounds) {
            const position = {
              x: event.clientX - reactFlowBounds.left,
              y: event.clientY - reactFlowBounds.top,
            };
            console.log("Drop position:", position);

            // Create a new node
            const newNode: ModelNode = {
              id: `${component.type || "layer"}-${Date.now()}`,
              type: component.type || "layer",
              name: component.name || "New Node",
              position: position,
              inputs: component.type === "input" ? [] : ["input"],
              outputs: component.type === "output" ? [] : ["output"],
              config: {},
            };

            console.log("Creating new node:", newNode);
            // Add the node to the canvas
            onNodeAdd(newNode);
          }
        } catch (error) {
          console.error("Error adding node:", error);
        }
      } else {
        console.warn("No component data found in the drop event");
      }
    },
    [onNodeAdd],
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
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-right"
        deleteKeyCode={["Backspace", "Delete"]}
        defaultEdgeOptions={{ type: "smoothstep" }}
        connectionLineType="smoothstep"
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
