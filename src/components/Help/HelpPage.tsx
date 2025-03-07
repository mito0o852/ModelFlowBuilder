import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Search,
  BookOpen,
  FileText,
  Video,
  MessageCircle,
  HelpCircle,
  ExternalLink,
  ChevronRight,
} from "lucide-react";

const HelpPage = () => {
  const [searchQuery, setSearchQuery] = useState("");

  // Sample FAQ data
  const faqs = [
    {
      question: "How do I create a new neural network model?",
      answer:
        "To create a new model, click on the 'New Project' button on the dashboard or projects page. This will open the model builder interface where you can drag and drop components to design your neural network architecture.",
    },
    {
      question: "Can I export my model to PyTorch code?",
      answer:
        "Yes, you can export your model to PyTorch code by clicking the 'Export Code' button in the top toolbar of the model builder. This will generate Python code that you can download and use in your own projects.",
    },
    {
      question: "How do I connect layers in my neural network?",
      answer:
        "To connect layers, click and drag from the output handle of one node to the input handle of another node. Valid connections will be highlighted in blue, while invalid connections will be shown in red.",
    },
    {
      question: "What types of layers are supported?",
      answer:
        "We support a wide range of layers including Linear, Convolutional (Conv1d, Conv2d), Recurrent (LSTM, GRU), Pooling, Normalization, and various activation functions like ReLU, Sigmoid, and Tanh.",
    },
    {
      question: "How do I test my model?",
      answer:
        "You can test your model by clicking the 'Test' button in the toolbar. This will open the test panel where you can provide sample input data and see how your model processes it.",
    },
    {
      question: "Can I save my model and continue working on it later?",
      answer:
        "Yes, you can save your model by clicking the 'Save' button in the toolbar. Your model will be saved to your account and you can access it later from the dashboard or projects page.",
    },
  ];

  // Sample tutorials data
  const tutorials = [
    {
      title: "Getting Started with Neural Network Builder",
      description:
        "Learn the basics of creating and configuring neural networks",
      duration: "5 min",
      thumbnail:
        "https://images.unsplash.com/photo-1591453089816-0fbb971b454c?w=500&q=80",
      link: "/tutorials/getting-started",
    },
    {
      title: "Building a Convolutional Neural Network",
      description:
        "Step-by-step guide to creating a CNN for image classification",
      duration: "10 min",
      thumbnail:
        "https://images.unsplash.com/photo-1561736778-92e52a7769ef?w=500&q=80",
      link: "/tutorials/cnn",
    },
    {
      title: "Working with Recurrent Neural Networks",
      description: "Learn how to build and configure RNNs for sequence data",
      duration: "12 min",
      thumbnail:
        "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=500&q=80",
      link: "/tutorials/rnn",
    },
    {
      title: "Advanced Model Configuration",
      description: "Deep dive into advanced settings and optimizations",
      duration: "15 min",
      thumbnail:
        "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=500&q=80",
      link: "/tutorials/advanced",
    },
  ];

  // Sample documentation sections
  const docSections = [
    {
      title: "Getting Started",
      items: [
        { title: "Introduction", link: "/docs/intro" },
        { title: "Installation", link: "/docs/installation" },
        { title: "Quick Start Guide", link: "/docs/quickstart" },
      ],
    },
    {
      title: "Core Concepts",
      items: [
        { title: "Neural Network Basics", link: "/docs/nn-basics" },
        { title: "Layer Types", link: "/docs/layers" },
        { title: "Activation Functions", link: "/docs/activations" },
        { title: "Model Architecture", link: "/docs/architecture" },
      ],
    },
    {
      title: "Features",
      items: [
        { title: "Model Builder", link: "/docs/model-builder" },
        { title: "Code Export", link: "/docs/code-export" },
        { title: "Testing & Validation", link: "/docs/testing" },
        { title: "Saving & Loading", link: "/docs/save-load" },
      ],
    },
    {
      title: "Advanced Topics",
      items: [
        { title: "Custom Layers", link: "/docs/custom-layers" },
        { title: "Performance Optimization", link: "/docs/optimization" },
        { title: "Deployment", link: "/docs/deployment" },
        { title: "API Reference", link: "/docs/api" },
      ],
    },
  ];

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Help Center</h1>
        <p className="text-muted-foreground">
          Find answers, tutorials, and documentation
        </p>
      </div>

      <div className="relative max-w-2xl mx-auto mb-10">
        <Search className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
        <Input
          placeholder="Search for help, tutorials, or documentation..."
          className="pl-10 py-6 text-lg"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <Tabs defaultValue="faq" className="w-full">
        <TabsList className="w-full max-w-md mx-auto grid grid-cols-3 mb-8">
          <TabsTrigger value="faq" className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            FAQ
          </TabsTrigger>
          <TabsTrigger value="tutorials" className="flex items-center gap-2">
            <Video className="h-4 w-4" />
            Tutorials
          </TabsTrigger>
          <TabsTrigger
            value="documentation"
            className="flex items-center gap-2"
          >
            <FileText className="h-4 w-4" />
            Documentation
          </TabsTrigger>
        </TabsList>

        <TabsContent value="faq" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>Frequently Asked Questions</CardTitle>
            </CardHeader>
            <CardContent>
              <Accordion type="single" collapsible className="w-full">
                {faqs.map((faq, index) => (
                  <AccordionItem key={index} value={`item-${index}`}>
                    <AccordionTrigger className="text-left">
                      {faq.question}
                    </AccordionTrigger>
                    <AccordionContent className="text-muted-foreground">
                      {faq.answer}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>

              <div className="mt-8 p-4 bg-muted rounded-lg">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  <MessageCircle className="h-5 w-5" />
                  Still have questions?
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Can't find the answer you're looking for? Please contact our
                  support team.
                </p>
                <Button>Contact Support</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tutorials" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>Video Tutorials</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {tutorials.map((tutorial, index) => (
                  <Card key={index} className="overflow-hidden">
                    <div className="h-48 overflow-hidden">
                      <img
                        src={tutorial.thumbnail}
                        alt={tutorial.title}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <CardContent className="p-4">
                      <h3 className="font-medium mb-1">{tutorial.title}</h3>
                      <p className="text-sm text-muted-foreground mb-2">
                        {tutorial.description}
                      </p>
                      <div className="flex justify-between items-center">
                        <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded">
                          {tutorial.duration}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="flex items-center gap-1"
                          asChild
                        >
                          <a href={tutorial.link}>
                            Watch
                            <ExternalLink className="h-3 w-3 ml-1" />
                          </a>
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <div className="mt-8 text-center">
                <Button variant="outline" className="flex items-center gap-2">
                  View All Tutorials
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="documentation" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Documentation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {docSections.map((section, index) => (
                  <div key={index}>
                    <h3 className="font-medium text-lg mb-4">
                      {section.title}
                    </h3>
                    <ScrollArea className="h-[200px] pr-4">
                      <div className="space-y-2">
                        {section.items.map((item, itemIndex) => (
                          <React.Fragment key={itemIndex}>
                            <a
                              href={item.link}
                              className="flex items-center justify-between p-2 hover:bg-muted rounded-md transition-colors"
                            >
                              <span>{item.title}</span>
                              <ChevronRight className="h-4 w-4 text-muted-foreground" />
                            </a>
                            {itemIndex < section.items.length - 1 && (
                              <Separator />
                            )}
                          </React.Fragment>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                ))}
              </div>

              <div className="mt-8 p-4 bg-muted rounded-lg">
                <h3 className="font-medium mb-2">API Documentation</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Looking for detailed API documentation? Check out our
                  comprehensive API reference.
                </p>
                <Button variant="outline" className="flex items-center gap-2">
                  View API Docs
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default HelpPage;
