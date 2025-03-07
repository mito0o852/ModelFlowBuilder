import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Layers, Activity, GitBranch, Info, Plus } from "lucide-react";

interface ComponentCardProps {
  name: string;
  description?: string;
  icon?: "layer" | "activation" | "operation" | "custom";
  onClick?: () => void;
  className?: string;
}

const ComponentCard = ({
  name = "Linear Layer",
  description = "A fully connected neural network layer",
  icon = "layer",
  onClick = () => {},
  className = "",
}: ComponentCardProps) => {
  // Map icon type to the appropriate Lucide icon
  const IconComponent = {
    layer: Layers,
    activation: Activity,
    operation: GitBranch,
    custom: Info,
  }[icon];

  return (
    <Card
      className={cn(
        "cursor-grab active:cursor-grabbing bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors",
        className,
      )}
      onClick={onClick}
      draggable="true"
      onDragStart={(e) => {
        // Set data for drag operation
        e.dataTransfer.effectAllowed = "move";
        const data = JSON.stringify({ type: icon, name, description });
        e.dataTransfer.setData("application/json", data);
        console.log("Drag started with data:", data);

        // Set a drag image
        const dragImage = document.createElement("div");
        dragImage.textContent = name;
        dragImage.className =
          "bg-primary text-primary-foreground p-2 rounded text-sm";
        document.body.appendChild(dragImage);
        e.dataTransfer.setDragImage(dragImage, 0, 0);
        setTimeout(() => document.body.removeChild(dragImage), 0);
      }}
    >
      <CardContent className="flex items-center p-4 h-full">
        <div className="mr-3 p-2 rounded-md bg-primary/10 dark:bg-primary/20">
          {IconComponent && (
            <IconComponent className="h-5 w-5 text-primary dark:text-primary-foreground" />
          )}
        </div>
        <div className="flex-1 overflow-hidden">
          <h3 className="font-medium text-sm truncate max-w-full text-foreground">
            {name}
          </h3>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <p className="text-xs text-muted-foreground line-clamp-1 max-w-full">
                  {description}
                </p>
              </TooltipTrigger>
              <TooltipContent side="right" className="max-w-[300px]">
                <p>{description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="ml-1 h-7 w-7 text-primary hover:bg-primary/10"
                onClick={(e) => {
                  e.stopPropagation();
                  onClick();
                }}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Add to canvas</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </CardContent>
    </Card>
  );
};

export default ComponentCard;
