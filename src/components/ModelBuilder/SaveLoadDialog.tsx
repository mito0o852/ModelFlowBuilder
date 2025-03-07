import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Save, FolderOpen, Trash2 } from "lucide-react";

interface SavedModel {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  lastModified: string;
}

interface SaveLoadDialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onSave?: (name: string, description: string) => void;
  onLoad?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  savedModels?: SavedModel[];
}

const SaveLoadDialog = ({
  open = true,
  onOpenChange = () => {},
  onSave = () => {},
  onLoad = () => {},
  onDelete = () => {},
  savedModels = [
    {
      id: "1",
      name: "Simple CNN",
      description:
        "A basic convolutional neural network for image classification",
      createdAt: "2023-06-15T10:30:00Z",
      lastModified: "2023-06-16T14:20:00Z",
    },
    {
      id: "2",
      name: "LSTM Sequence Model",
      description: "Long short-term memory network for sequence prediction",
      createdAt: "2023-07-01T09:15:00Z",
      lastModified: "2023-07-02T11:45:00Z",
    },
    {
      id: "3",
      name: "Transformer Encoder",
      description: "Transformer-based encoder architecture",
      createdAt: "2023-07-10T16:20:00Z",
      lastModified: "2023-07-10T16:20:00Z",
    },
  ],
}: SaveLoadDialogProps) => {
  const [activeTab, setActiveTab] = useState<"save" | "load">("save");
  const [modelName, setModelName] = useState("");
  const [modelDescription, setModelDescription] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const handleSave = () => {
    if (modelName.trim()) {
      onSave(modelName, modelDescription);
      setModelName("");
      setModelDescription("");
      onOpenChange(false);
    }
  };

  const handleLoad = () => {
    if (selectedModelId) {
      onLoad(selectedModelId);
      setSelectedModelId(null);
      onOpenChange(false);
    }
  };

  const handleDelete = (id: string) => {
    onDelete(id);
    if (selectedModelId === id) {
      setSelectedModelId(null);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-white sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Save or Load Model</DialogTitle>
          <DialogDescription>
            Save your current model or load a previously saved one.
          </DialogDescription>
        </DialogHeader>

        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as "save" | "load")}
          className="w-full"
        >
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="save" className="flex items-center gap-2">
              <Save className="h-4 w-4" />
              Save Model
            </TabsTrigger>
            <TabsTrigger value="load" className="flex items-center gap-2">
              <FolderOpen className="h-4 w-4" />
              Load Model
            </TabsTrigger>
          </TabsList>

          <TabsContent value="save" className="space-y-4 py-4">
            <div className="space-y-2">
              <label htmlFor="model-name" className="text-sm font-medium">
                Model Name
              </label>
              <Input
                id="model-name"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="Enter model name"
              />
            </div>
            <div className="space-y-2">
              <label
                htmlFor="model-description"
                className="text-sm font-medium"
              >
                Description (optional)
              </label>
              <Input
                id="model-description"
                value={modelDescription}
                onChange={(e) => setModelDescription(e.target.value)}
                placeholder="Enter model description"
              />
            </div>
          </TabsContent>

          <TabsContent value="load" className="py-4">
            <div className="max-h-[250px] overflow-y-auto space-y-2 pr-2">
              {savedModels.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  No saved models found
                </p>
              ) : (
                savedModels.map((model) => (
                  <div
                    key={model.id}
                    className={`p-3 border rounded-md cursor-pointer transition-colors ${selectedModelId === model.id ? "border-primary bg-primary/5" : "hover:bg-gray-50"}`}
                    onClick={() => setSelectedModelId(model.id)}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium">{model.name}</h3>
                        {model.description && (
                          <p className="text-sm text-muted-foreground mt-1">
                            {model.description}
                          </p>
                        )}
                        <p className="text-xs text-muted-foreground mt-2">
                          Last modified: {formatDate(model.lastModified)}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(model.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          {activeTab === "save" ? (
            <Button onClick={handleSave} disabled={!modelName.trim()}>
              Save
            </Button>
          ) : (
            <Button onClick={handleLoad} disabled={!selectedModelId}>
              Load
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default SaveLoadDialog;
