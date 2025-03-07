import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import {
  Save,
  FolderOpen,
  Code,
  Download,
  Play,
  ChevronDown,
  Settings,
  HelpCircle,
} from "lucide-react";
import ThemeToggle from "./ThemeToggle";

interface HeaderProps {
  onSave?: () => void;
  onLoad?: () => void;
  onExport?: () => void;
  onTest?: () => void;
  onSettings?: () => void;
  onHelp?: () => void;
  className?: string;
}

const Header = ({
  onSave = () => {},
  onLoad = () => {},
  onExport = () => {},
  onTest = () => {},
  onSettings = () => {},
  onHelp = () => {},
  className = "",
}: HeaderProps) => {
  const [projectName, setProjectName] = useState("Untitled Neural Network");

  return (
    <header
      className={cn(
        "w-full h-16 px-4 flex items-center justify-between border-b bg-background z-30",
        className,
      )}
    >
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-semibold">Neural Network Builder</h1>
        <div className="h-6 w-px bg-border mx-2" />
        <div className="relative">
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            className="h-8 px-2 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-primary bg-background text-foreground"
            aria-label="Project name"
          />
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Button
          variant="outline"
          size="sm"
          onClick={onSave}
          className="flex items-center"
        >
          <Save className="h-4 w-4 mr-2" />
          Save
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={onLoad}
          className="flex items-center"
        >
          <FolderOpen className="h-4 w-4 mr-2" />
          Load
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={onTest}
          className="flex items-center"
        >
          <Play className="h-4 w-4 mr-2" />
          Test
        </Button>

        <Button
          variant="primary"
          size="sm"
          onClick={onExport}
          className="flex items-center"
        >
          <Code className="h-4 w-4 mr-2" />
          Export Code
        </Button>

        <ThemeToggle />

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon">
              <ChevronDown className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={onSettings}>
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={onHelp}>
              <HelpCircle className="h-4 w-4 mr-2" />
              Help & Documentation
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => window.open("/", "_blank")}>
              <Download className="h-4 w-4 mr-2" />
              Download Model
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
};

export default Header;
