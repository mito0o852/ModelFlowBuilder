import React from "react";
import { cn } from "@/lib/utils";
import {
  Search,
  Home,
  Box,
  FileText,
  Layers,
  Bell,
  HelpCircle,
  Settings,
  Sliders,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SidebarProps {
  activeTab?: string;
  onTabChange?: (tab: string) => void;
  className?: string;
}

const Sidebar = ({
  activeTab = "components",
  onTabChange = () => {},
  className = "",
}: SidebarProps) => {
  const tabs = [
    { id: "search", icon: Search, label: "Search" },
    { id: "home", icon: Home, label: "Home" },
    { id: "components", icon: Box, label: "Components" },
    { id: "documents", icon: FileText, label: "Documents" },
    { id: "layers", icon: Layers, label: "Layers" },
    { id: "sliders", icon: Sliders, label: "Settings" },
  ];

  const bottomTabs = [
    { id: "notifications", icon: Bell, label: "Notifications" },
    { id: "help", icon: HelpCircle, label: "Help" },
    { id: "settings", icon: Settings, label: "Settings" },
  ];

  return (
    <div
      className={cn(
        "flex flex-col justify-between h-full w-16 bg-background dark:bg-gray-900 text-foreground dark:text-white py-4 border-r border-border",
        className,
      )}
    >
      <div className="flex flex-col items-center space-y-6">
        <div className="mb-6">
          <div className="w-8 h-8 rounded-full bg-primary/20 dark:bg-indigo-400 flex items-center justify-center">
            <Box className="h-5 w-5 text-gray-900 dark:text-gray-100" />
          </div>
        </div>

        <div className="flex flex-col items-center space-y-4 w-full">
          {tabs.map((tab) => (
            <TooltipProvider key={tab.id}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    className={cn(
                      "w-full flex justify-center py-2 relative",
                      activeTab === tab.id
                        ? "text-white"
                        : "text-gray-400 hover:text-gray-200",
                    )}
                    onClick={() => onTabChange(tab.id)}
                  >
                    {activeTab === tab.id && (
                      <div className="absolute right-0 w-1 h-8 bg-white rounded-l-md" />
                    )}
                    <tab.icon className="h-5 w-5" />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <p>{tab.label}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ))}
        </div>
      </div>

      <div className="flex flex-col items-center space-y-4">
        {bottomTabs.map((tab) => (
          <TooltipProvider key={tab.id}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className={cn(
                    "w-full flex justify-center py-2 relative",
                    activeTab === tab.id
                      ? "text-primary dark:text-white"
                      : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200",
                  )}
                  onClick={() => onTabChange(tab.id)}
                >
                  {activeTab === tab.id && (
                    <div className="absolute right-0 w-1 h-8 bg-white rounded-l-md" />
                  )}
                  <tab.icon className="h-5 w-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>{tab.label}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}

        <div className="mt-2 flex justify-center">
          <div className="relative">
            <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center overflow-hidden">
              <img
                src="https://api.dicebear.com/7.x/avataaars/svg?seed=user"
                alt="User"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-green-500 border-2 border-gray-900" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
