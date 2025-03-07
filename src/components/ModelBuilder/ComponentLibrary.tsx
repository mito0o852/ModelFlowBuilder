import React, { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Search, Layers, Activity, GitBranch, Settings } from "lucide-react";
import ComponentCard from "./ComponentCard";

interface ComponentLibraryProps {
  onComponentSelect?: (component: {
    type: string;
    name: string;
    category: string;
  }) => void;
}

const ComponentLibrary = ({
  onComponentSelect = () => {},
}: ComponentLibraryProps) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("layers");

  // Define component categories and their items
  const componentCategories = {
    layers: [
      {
        name: "Linear",
        description: "A fully connected neural network layer",
        icon: "layer" as const,
      },
      {
        name: "Conv2d",
        description: "2D convolutional layer for image processing",
        icon: "layer" as const,
      },
      {
        name: "LSTM",
        description: "Long Short-Term Memory recurrent layer",
        icon: "layer" as const,
      },
      {
        name: "Embedding",
        description: "Converts indices to dense vectors",
        icon: "layer" as const,
      },
      {
        name: "BatchNorm",
        description: "Batch normalization layer",
        icon: "layer" as const,
      },
    ],
    activations: [
      {
        name: "ReLU",
        description: "Rectified Linear Unit activation function",
        icon: "activation" as const,
      },
      {
        name: "Sigmoid",
        description: "Sigmoid activation function",
        icon: "activation" as const,
      },
      {
        name: "Tanh",
        description: "Hyperbolic tangent activation function",
        icon: "activation" as const,
      },
      {
        name: "Softmax",
        description: "Softmax activation for classification",
        icon: "activation" as const,
      },
      {
        name: "LeakyReLU",
        description: "Leaky version of ReLU with small slope",
        icon: "activation" as const,
      },
    ],
    operations: [
      {
        name: "Concat",
        description: "Concatenate tensors along dimension",
        icon: "operation" as const,
      },
      {
        name: "Add",
        description: "Element-wise addition of tensors",
        icon: "operation" as const,
      },
      {
        name: "Multiply",
        description: "Element-wise multiplication of tensors",
        icon: "operation" as const,
      },
      {
        name: "Flatten",
        description: "Flatten dimensions into a 1D tensor",
        icon: "operation" as const,
      },
      {
        name: "Reshape",
        description: "Reshape tensor to new dimensions",
        icon: "operation" as const,
      },
    ],
    custom: [
      {
        name: "Custom Layer",
        description: "Define your own custom layer",
        icon: "custom" as const,
      },
      {
        name: "Import Module",
        description: "Import an existing PyTorch module",
        icon: "custom" as const,
      },
    ],
  };

  // Filter components based on search query
  const filterComponents = (components: any[]) => {
    if (!searchQuery) return components;
    return components.filter(
      (comp) =>
        comp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        comp.description.toLowerCase().includes(searchQuery.toLowerCase()),
    );
  };

  // Handle component selection
  const handleComponentSelect = (component: any) => {
    onComponentSelect({
      ...component,
      category: activeTab,
    });
  };

  return (
    <div className="w-[280px] h-[800px] border-r border-border bg-background flex flex-col z-10">
      <div className="p-4 border-b border-border">
        <h2 className="text-lg font-semibold mb-2">Component Library</h2>
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search components..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <Tabs
        defaultValue="layers"
        value={activeTab}
        onValueChange={setActiveTab}
        className="flex-1 flex flex-col"
      >
        <TabsList className="grid grid-cols-4 h-auto">
          <TabsTrigger
            value="layers"
            className="flex flex-col items-center py-2 px-1 h-auto"
          >
            <Layers className="h-4 w-4 mb-1" />
            <span className="text-xs whitespace-nowrap">Layers</span>
          </TabsTrigger>
          <TabsTrigger
            value="activations"
            className="flex flex-col items-center py-2 px-1 h-auto"
          >
            <Activity className="h-4 w-4 mb-1" />
            <span className="text-xs whitespace-nowrap">Activations</span>
          </TabsTrigger>
          <TabsTrigger
            value="operations"
            className="flex flex-col items-center py-2 px-1 h-auto"
          >
            <GitBranch className="h-4 w-4 mb-1" />
            <span className="text-xs whitespace-nowrap">Operations</span>
          </TabsTrigger>
          <TabsTrigger
            value="custom"
            className="flex flex-col items-center py-2 px-1 h-auto"
          >
            <Settings className="h-4 w-4 mb-1" />
            <span className="text-xs whitespace-nowrap">Custom</span>
          </TabsTrigger>
        </TabsList>

        <div className="flex-1 overflow-hidden">
          {Object.entries(componentCategories).map(([category, components]) => (
            <TabsContent
              key={category}
              value={category}
              className="h-full mt-0 p-0"
            >
              <ScrollArea className="h-[calc(100vh-250px)] px-4 py-2">
                <div className="space-y-3">
                  {filterComponents(components).length > 0 ? (
                    filterComponents(components).map((component, index) => (
                      <div key={index}>
                        <ComponentCard
                          name={component.name}
                          description={component.description}
                          icon={component.icon}
                          onClick={() => handleComponentSelect(component)}
                        />
                        {index < filterComponents(components).length - 1 && (
                          <Separator className="my-3" />
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      No components found
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>
          ))}
        </div>
      </Tabs>

      <div className="p-3 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          Drag components to the canvas to build your neural network
        </p>
      </div>
    </div>
  );
};

export default ComponentLibrary;
