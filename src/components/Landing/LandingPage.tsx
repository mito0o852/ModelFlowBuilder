import React from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  ArrowRight,
  Brain,
  Layers,
  Code,
  Zap,
  Github,
  Twitter,
} from "lucide-react";

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Header/Navigation */}
      <header className="container mx-auto py-6 px-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          <span className="text-xl font-bold">NeuralBuilder</span>
        </div>
        <nav className="hidden md:flex items-center gap-8">
          <a
            href="#features"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Features
          </a>
          <a
            href="#how-it-works"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            How It Works
          </a>
          <a
            href="#testimonials"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Testimonials
          </a>
          <Link
            to="/help"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Documentation
          </Link>
        </nav>
        <div className="flex items-center gap-4">
          <Button variant="outline" asChild>
            <Link to="/login">Log In</Link>
          </Button>
          <Button asChild>
            <Link to="/signup">Sign Up</Link>
          </Button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 md:py-32 flex flex-col md:flex-row items-center gap-12">
        <div className="flex-1 space-y-6">
          <h1 className="text-4xl md:text-6xl font-bold leading-tight">
            Build Neural Networks
            <span className="text-primary"> Visually</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-xl">
            Design, test, and export deep learning models with an intuitive
            drag-and-drop interface. No coding required.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 pt-4">
            <Button size="lg" asChild>
              <Link to="/signup" className="flex items-center gap-2">
                Get Started
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link to="/demo">Try Demo</Link>
            </Button>
          </div>
        </div>
        <div className="flex-1">
          <div className="relative">
            <div className="absolute -top-6 -left-6 w-full h-full rounded-xl bg-primary/10 z-0"></div>
            <img
              src="https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=800&q=80"
              alt="Neural Network Builder Interface"
              className="rounded-xl shadow-xl z-10 relative"
            />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-muted/50 py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">Powerful Features</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Everything you need to design, test, and implement neural networks
              without writing a single line of code.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="p-3 rounded-full bg-primary/10 w-fit">
                  <Layers className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-medium">Drag & Drop Builder</h3>
                <p className="text-muted-foreground">
                  Intuitively design neural networks by dragging and connecting
                  components on a visual canvas.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="p-3 rounded-full bg-primary/10 w-fit">
                  <Zap className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-medium">Real-time Testing</h3>
                <p className="text-muted-foreground">
                  Test your models with sample data and visualize the results
                  instantly.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardContent className="p-6 space-y-4">
                <div className="p-3 rounded-full bg-primary/10 w-fit">
                  <Code className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-medium">Code Export</h3>
                <p className="text-muted-foreground">
                  Generate clean, optimized PyTorch code from your visual
                  designs with one click.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">How It Works</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Building neural networks has never been easier. Follow these
              simple steps to create your model.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center space-y-4">
              <div className="bg-primary/10 rounded-full w-12 h-12 flex items-center justify-center mx-auto">
                <span className="text-primary font-bold">1</span>
              </div>
              <h3 className="text-xl font-medium">Design</h3>
              <p className="text-muted-foreground">
                Drag components from the library and connect them to create your
                neural network architecture.
              </p>
            </div>

            <div className="text-center space-y-4">
              <div className="bg-primary/10 rounded-full w-12 h-12 flex items-center justify-center mx-auto">
                <span className="text-primary font-bold">2</span>
              </div>
              <h3 className="text-xl font-medium">Configure</h3>
              <p className="text-muted-foreground">
                Set parameters for each layer and component to customize your
                model's behavior.
              </p>
            </div>

            <div className="text-center space-y-4">
              <div className="bg-primary/10 rounded-full w-12 h-12 flex items-center justify-center mx-auto">
                <span className="text-primary font-bold">3</span>
              </div>
              <h3 className="text-xl font-medium">Export</h3>
              <p className="text-muted-foreground">
                Test your model with sample data and export the code to use in
                your projects.
              </p>
            </div>
          </div>

          <div className="mt-16 text-center">
            <Button size="lg" asChild>
              <Link to="/signup">Start Building Now</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="bg-muted/50 py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold mb-4">What Users Say</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Hear from researchers and developers who use our platform to build
              neural networks.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="bg-card">
              <CardContent className="p-6 space-y-4">
                <p className="italic text-muted-foreground">
                  "This tool has completely transformed how I prototype neural
                  networks. What used to take hours now takes minutes."
                </p>
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full overflow-hidden">
                    <img
                      src="https://api.dicebear.com/7.x/avataaars/svg?seed=John"
                      alt="John D."
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div>
                    <p className="font-medium">John D.</p>
                    <p className="text-sm text-muted-foreground">
                      ML Researcher
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardContent className="p-6 space-y-4">
                <p className="italic text-muted-foreground">
                  "As someone new to deep learning, this visual approach helped
                  me understand neural networks in a way textbooks couldn't."
                </p>
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full overflow-hidden">
                    <img
                      src="https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah"
                      alt="Sarah M."
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div>
                    <p className="font-medium">Sarah M.</p>
                    <p className="text-sm text-muted-foreground">Student</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardContent className="p-6 space-y-4">
                <p className="italic text-muted-foreground">
                  "The code export feature is a game-changer. I design visually
                  and get clean, production-ready PyTorch code."
                </p>
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full overflow-hidden">
                    <img
                      src="https://api.dicebear.com/7.x/avataaars/svg?seed=Michael"
                      alt="Michael R."
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div>
                    <p className="font-medium">Michael R.</p>
                    <p className="text-sm text-muted-foreground">
                      Software Engineer
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4 max-w-4xl">
          <div className="bg-primary/10 rounded-2xl p-8 md:p-12 text-center">
            <h2 className="text-3xl font-bold mb-4">
              Ready to Build Your Neural Network?
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto mb-8">
              Join thousands of researchers and developers who are creating
              neural networks visually.
            </p>
            <Button
              size="lg"
              className="bg-primary text-primary-foreground"
              asChild
            >
              <Link to="/signup">Get Started for Free</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-muted/50 py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Brain className="h-6 w-6 text-primary" />
                <span className="text-lg font-bold">NeuralBuilder</span>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Building neural networks visually, without code.
              </p>
              <div className="flex gap-4">
                <a
                  href="#"
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Github className="h-5 w-5" />
                </a>
                <a
                  href="#"
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Twitter className="h-5 w-5" />
                </a>
              </div>
            </div>

            <div>
              <h3 className="font-medium mb-4">Product</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Features
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Pricing
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Testimonials
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    FAQ
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-4">Resources</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Documentation
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Tutorials
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Blog
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Support
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-4">Company</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    About
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Careers
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    className="hover:text-foreground transition-colors"
                  >
                    Terms of Service
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-border pt-8 text-center text-sm text-muted-foreground">
            <p>
              © {new Date().getFullYear()} NeuralBuilder. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
