import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Plus, Layers, Activity, GitBranch, Clock } from "lucide-react";
import { Link } from "react-router-dom";

const Dashboard = () => {
  // Sample data for recent projects
  const recentProjects = [
    {
      id: "1",
      name: "Simple CNN",
      description:
        "A basic convolutional neural network for image classification",
      lastModified: "2023-06-16T14:20:00Z",
    },
    {
      id: "2",
      name: "LSTM Sequence Model",
      description: "Long short-term memory network for sequence prediction",
      lastModified: "2023-07-02T11:45:00Z",
    },
    {
      id: "3",
      name: "Transformer Encoder",
      description: "Transformer-based encoder architecture",
      lastModified: "2023-07-10T16:20:00Z",
    },
  ];

  // Sample data for statistics
  const stats = [
    { name: "Total Projects", value: 12, icon: Layers },
    { name: "Models Created", value: 28, icon: Activity },
    { name: "Code Exports", value: 15, icon: GitBranch },
  ];

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
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <Button asChild>
          <Link to="/projects/new" className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            New Project
          </Link>
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardContent className="flex items-center p-6">
              <div className="p-4 rounded-full bg-primary/10 mr-4">
                <stat.icon className="h-6 w-6 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{stat.name}</p>
                <h3 className="text-2xl font-bold">{stat.value}</h3>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Recent Projects */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Recent Projects</h2>
          <Button variant="outline" asChild>
            <Link to="/projects">View All</Link>
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {recentProjects.map((project) => (
            <Card
              key={project.id}
              className="hover:shadow-md transition-shadow"
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">{project.name}</CardTitle>
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
          <Card className="border-dashed flex items-center justify-center h-[200px] hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors cursor-pointer">
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
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <Button
            variant="outline"
            className="h-auto py-4 flex flex-col items-center justify-center gap-2"
            asChild
          >
            <Link to="/projects/templates">
              <Layers className="h-5 w-5 mb-1" />
              <span>Browse Templates</span>
            </Link>
          </Button>
          <Button
            variant="outline"
            className="h-auto py-4 flex flex-col items-center justify-center gap-2"
            asChild
          >
            <Link to="/help/tutorials">
              <Activity className="h-5 w-5 mb-1" />
              <span>Tutorials</span>
            </Link>
          </Button>
          <Button
            variant="outline"
            className="h-auto py-4 flex flex-col items-center justify-center gap-2"
            asChild
          >
            <Link to="/help/documentation">
              <GitBranch className="h-5 w-5 mb-1" />
              <span>Documentation</span>
            </Link>
          </Button>
          <Button
            variant="outline"
            className="h-auto py-4 flex flex-col items-center justify-center gap-2"
            asChild
          >
            <Link to="/settings">
              <Clock className="h-5 w-5 mb-1" />
              <span>Settings</span>
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
