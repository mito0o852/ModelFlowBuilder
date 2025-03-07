import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Plus,
  Search,
  Clock,
  MoreVertical,
  Edit,
  Copy,
  Trash2,
  Download,
  Grid,
  List,
} from "lucide-react";

interface Project {
  id: string;
  name: string;
  description: string;
  lastModified: string;
  createdAt: string;
  thumbnail?: string;
}

const ProjectsList = () => {
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("all");

  // Sample projects data
  const allProjects: Project[] = [
    {
      id: "1",
      name: "Simple CNN",
      description:
        "A basic convolutional neural network for image classification",
      lastModified: "2023-06-16T14:20:00Z",
      createdAt: "2023-06-15T10:30:00Z",
      thumbnail:
        "https://images.unsplash.com/photo-1677442135136-760c813a6f14?w=500&q=80",
    },
    {
      id: "2",
      name: "LSTM Sequence Model",
      description: "Long short-term memory network for sequence prediction",
      lastModified: "2023-07-02T11:45:00Z",
      createdAt: "2023-07-01T09:15:00Z",
    },
    {
      id: "3",
      name: "Transformer Encoder",
      description: "Transformer-based encoder architecture",
      lastModified: "2023-07-10T16:20:00Z",
      createdAt: "2023-07-10T16:20:00Z",
      thumbnail:
        "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=500&q=80",
    },
    {
      id: "4",
      name: "GAN for Image Generation",
      description:
        "Generative adversarial network for creating realistic images",
      lastModified: "2023-08-05T13:10:00Z",
      createdAt: "2023-08-01T11:30:00Z",
    },
    {
      id: "5",
      name: "ResNet Implementation",
      description:
        "Implementation of ResNet architecture with skip connections",
      lastModified: "2023-08-12T09:45:00Z",
      createdAt: "2023-08-10T14:20:00Z",
      thumbnail:
        "https://images.unsplash.com/photo-1542281286-9e0a16bb7366?w=500&q=80",
    },
    {
      id: "6",
      name: "Autoencoder for Dimensionality Reduction",
      description: "Neural network for unsupervised feature learning",
      lastModified: "2023-08-18T15:30:00Z",
      createdAt: "2023-08-15T10:15:00Z",
    },
  ];

  // Filter projects based on search query and active tab
  const filteredProjects = allProjects
    .filter(
      (project) =>
        project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.description.toLowerCase().includes(searchQuery.toLowerCase()),
    )
    .sort(
      (a, b) =>
        new Date(b.lastModified).getTime() - new Date(a.lastModified).getTime(),
    );

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Projects</h1>
        <Button asChild>
          <Link to="/projects/new" className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            New Project
          </Link>
        </Button>
      </div>

      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div className="relative w-full md:w-[300px]">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search projects..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="flex items-center gap-4 w-full md:w-auto">
          <Tabs
            defaultValue="all"
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full md:w-auto"
          >
            <TabsList>
              <TabsTrigger value="all">All Projects</TabsTrigger>
              <TabsTrigger value="recent">Recent</TabsTrigger>
              <TabsTrigger value="favorites">Favorites</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="flex border rounded-md">
            <Button
              variant={viewMode === "grid" ? "default" : "ghost"}
              size="icon"
              className="rounded-r-none"
              onClick={() => setViewMode("grid")}
            >
              <Grid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "list" ? "default" : "ghost"}
              size="icon"
              className="rounded-l-none"
              onClick={() => setViewMode("list")}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {filteredProjects.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No projects found</p>
          <Button asChild>
            <Link to="/projects/new">Create New Project</Link>
          </Button>
        </div>
      ) : viewMode === "grid" ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProjects.map((project) => (
            <Card
              key={project.id}
              className="overflow-hidden hover:shadow-md transition-shadow"
            >
              {project.thumbnail && (
                <div className="h-40 overflow-hidden">
                  <img
                    src={project.thumbnail}
                    alt={project.name}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}
              <CardHeader
                className={`pb-2 ${!project.thumbnail ? "pt-6" : ""}`}
              >
                <div className="flex justify-between items-start">
                  <CardTitle className="text-lg">{project.name}</CardTitle>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem asChild>
                        <Link
                          to={`/projects/${project.id}`}
                          className="flex items-center"
                        >
                          <Edit className="h-4 w-4 mr-2" /> Edit
                        </Link>
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <Copy className="h-4 w-4 mr-2" /> Duplicate
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <Download className="h-4 w-4 mr-2" /> Export
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-destructive">
                        <Trash2 className="h-4 w-4 mr-2" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                  {project.description}
                </p>
                <div className="flex items-center text-xs text-muted-foreground">
                  <Clock className="h-3 w-3 mr-1" />
                  Last modified: {formatDate(project.lastModified)}
                </div>
              </CardContent>
            </Card>
          ))}
          <Card className="border-dashed flex items-center justify-center h-[250px] hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors cursor-pointer">
            <Link to="/projects/new" className="flex flex-col items-center p-6">
              <div className="p-3 rounded-full bg-primary/10 mb-3">
                <Plus className="h-6 w-6 text-primary" />
              </div>
              <p className="font-medium">Create New Project</p>
              <p className="text-sm text-muted-foreground">
                Start building a new neural network
              </p>
            </Link>
          </Card>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredProjects.map((project) => (
            <Card
              key={project.id}
              className="hover:shadow-md transition-shadow"
            >
              <div className="flex items-center p-4">
                <div className="flex-1">
                  <h3 className="font-medium">{project.name}</h3>
                  <p className="text-sm text-muted-foreground line-clamp-1">
                    {project.description}
                  </p>
                </div>
                <div className="text-sm text-muted-foreground mr-4">
                  {formatDate(project.lastModified)}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" asChild>
                    <Link to={`/projects/${project.id}`}>Open</Link>
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem>
                        <Copy className="h-4 w-4 mr-2" /> Duplicate
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <Download className="h-4 w-4 mr-2" /> Export
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-destructive">
                        <Trash2 className="h-4 w-4 mr-2" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default ProjectsList;
