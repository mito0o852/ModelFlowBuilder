import React from "react";
import { Outlet, Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Brain,
  Home,
  FolderKanban,
  Settings,
  HelpCircle,
  Bell,
  Search,
  LogOut,
  User,
} from "lucide-react";

const MainLayout = () => {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b bg-background z-10">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link to="/" className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-primary" />
              <span className="font-bold text-lg">NeuralBuilder</span>
            </Link>

            <nav className="hidden md:flex items-center gap-6">
              <Link
                to="/dashboard"
                className={`flex items-center gap-2 text-sm ${isActive("/dashboard") ? "text-foreground font-medium" : "text-muted-foreground hover:text-foreground"}`}
              >
                <Home className="h-4 w-4" />
                Dashboard
              </Link>
              <Link
                to="/projects"
                className={`flex items-center gap-2 text-sm ${isActive("/projects") ? "text-foreground font-medium" : "text-muted-foreground hover:text-foreground"}`}
              >
                <FolderKanban className="h-4 w-4" />
                Projects
              </Link>
              <Link
                to="/help"
                className={`flex items-center gap-2 text-sm ${isActive("/help") ? "text-foreground font-medium" : "text-muted-foreground hover:text-foreground"}`}
              >
                <HelpCircle className="h-4 w-4" />
                Help
              </Link>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative hidden md:flex items-center">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search..."
                className="w-[200px] h-9 rounded-md border border-input bg-background px-8 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>

            <Button variant="ghost" size="icon" className="relative">
              <Bell className="h-5 w-5" />
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-primary"></span>
            </Button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-9 w-9 rounded-full">
                  <Avatar className="h-9 w-9">
                    <AvatarImage src="https://api.dicebear.com/7.x/avataaars/svg?seed=Alex" alt="User" />
                    <AvatarFallback>AJ</AvatarFallback>
