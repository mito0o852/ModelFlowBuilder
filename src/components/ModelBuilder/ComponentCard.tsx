import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Layers, Activity, GitBranch, Info } from "lucide-react";

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
        "w-[260px] h-[80px] cursor-grab active:cursor-grabbing bg-white hover:bg-gray-50 transition-colors",
        className,
      )}
      onClick={onClick}
      draggable
      onDragStart={(e) => {
        // Set data for drag operation
        e.dataTransfer.setData(
          "application/json",
          JSON.stringify({ type: icon, name }),
        );
        // Optional: set a drag image
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
        <div className="mr-3 p-2 rounded-md bg-primary/10">
          {IconComponent && <IconComponent className="h-5 w-5 text-primary" />}
        </div>
        <div className="flex-1 overflow-hidden">
          <h3 className="font-medium text-sm truncate">{name}</h3>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <p className="text-xs text-muted-foreground truncate">
                  {description}
                </p>
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-[200px]">{description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CardContent>
    </Card>
  );
};

export default ComponentCard;
