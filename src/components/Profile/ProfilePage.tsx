import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { PencilIcon, Save, User, Mail, Github, Twitter } from "lucide-react";

const ProfilePage = () => {
  // Sample user data
  const user = {
    name: "Alex Johnson",
    email: "alex.johnson@example.com",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Alex",
    bio: "Machine Learning Engineer with a focus on computer vision and neural networks. I love building and experimenting with different model architectures.",
    location: "San Francisco, CA",
    website: "https://alexjohnson.dev",
    github: "alexjohnson",
    twitter: "alexjohnson_ml",
    skills: [
      "PyTorch",
      "TensorFlow",
      "Computer Vision",
      "NLP",
      "GANs",
      "Reinforcement Learning",
    ],
    joinDate: "2022-03-15T00:00:00Z",
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <Card className="mb-8">
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row gap-6 items-start md:items-center">
            <Avatar className="h-24 w-24">
              <AvatarImage src={user.avatar} alt={user.name} />
              <AvatarFallback>{user.name.charAt(0)}</AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <h1 className="text-2xl font-bold mb-1">{user.name}</h1>
              <p className="text-muted-foreground mb-2">{user.email}</p>
              <p className="text-sm mb-3">{user.bio}</p>
              <div className="flex flex-wrap gap-2">
                {user.skills.map((skill, index) => (
                  <Badge key={index} variant="secondary">
                    {skill}
                  </Badge>
                ))}
              </div>
            </div>
            <Button variant="outline" className="flex items-center gap-2">
              <PencilIcon className="h-4 w-4" />
              Edit Profile
            </Button>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="profile" className="w-full">
        <TabsList className="w-full max-w-md mx-auto grid grid-cols-3 mb-8">
          <TabsTrigger value="profile">Profile</TabsTrigger>
          <TabsTrigger value="activity">Activity</TabsTrigger>
          <TabsTrigger value="projects">Projects</TabsTrigger>
        </TabsList>

        <TabsContent value="profile" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Personal Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input id="name" value={user.name} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" value={user.email} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="location">Location</Label>
                  <Input id="location" value={user.location} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="website">Website</Label>
                  <Input id="website" value={user.website} />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="bio">Bio</Label>
                <Input id="bio" value={user.bio} />
              </div>
              <Separator className="my-4" />
              <h3 className="text-lg font-medium mb-4">Social Profiles</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="github" className="flex items-center gap-2">
                    <Github className="h-4 w-4" /> GitHub
                  </Label>
                  <Input id="github" value={user.github} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="twitter" className="flex items-center gap-2">
                    <Twitter className="h-4 w-4" /> Twitter
                  </Label>
                  <Input id="twitter" value={user.twitter} />
                </div>
              </div>
              <div className="flex justify-end mt-6">
                <Button className="flex items-center gap-2">
                  <Save className="h-4 w-4" />
                  Save Changes
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Account Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h4 className="font-medium">Member Since</h4>
                    <p className="text-sm text-muted-foreground">
                      {formatDate(user.joinDate)}
                    </p>
                  </div>
                  <Badge>Active</Badge>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <div>
                    <h4 className="font-medium">Subscription Plan</h4>
                    <p className="text-sm text-muted-foreground">Free Tier</p>
                  </div>
                  <Button variant="outline">Upgrade</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="activity">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-8">
                {[1, 2, 3, 4].map((_, index) => (
                  <div key={index} className="flex gap-4">
                    <div className="min-w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                      <User className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium">
                        {index === 0
                          ? "Created a new project"
                          : index === 1
                            ? "Updated LSTM Sequence Model"
                            : index === 2
                              ? "Exported code for CNN model"
                              : "Shared a project with team"}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {index === 0
                          ? "2 hours ago"
                          : index === 1
                            ? "Yesterday"
                            : index === 2
                              ? "3 days ago"
                              : "1 week ago"}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="projects">
          <Card>
            <CardHeader>
              <CardTitle>Your Projects</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[1, 2, 3, 4].map((_, index) => (
                  <Card key={index} className="overflow-hidden">
                    <div className="h-32 bg-gray-100 dark:bg-gray-800">
                      {/* Project thumbnail would go here */}
                    </div>
                    <CardContent className="p-4">
                      <h3 className="font-medium">
                        {index === 0
                          ? "Simple CNN"
                          : index === 1
                            ? "LSTM Sequence Model"
                            : index === 2
                              ? "Transformer Encoder"
                              : "GAN for Image Generation"}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Last updated:{" "}
                        {index === 0
                          ? "2 days ago"
                          : index === 1
                            ? "1 week ago"
                            : index === 2
                              ? "2 weeks ago"
                              : "1 month ago"}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ProfilePage;
