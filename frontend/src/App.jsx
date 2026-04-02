import React from 'react';
import {
  Zap,
  Eye,
  Mic,
  Server,
  Cpu,
  Shield,
  ArrowRight,
  CheckCircle,
  Lightbulb,
} from 'lucide-react';
import { useNavigate, BrowserRouter, Routes, Route } from 'react-router-dom';

// Import your Upload page
import Upload from './Upload';
import Result from './Result';

// --- Data Definitions ---
const features = [
  {
    icon: Eye,
    title: "Visual Integrity Check",
    description: "Analyzes minute inconsistencies in facial expressions, eye movements, and lighting artifacts specific to visual deepfakes.",
  },
  {
    icon: Mic,
    title: "Acoustic Anomaly Detection",
    description: "Examines audio tracks for unnatural tones, speech patterns, and artifacts often present in synthesized voice or lip-sync errors.",
  },
  {
    icon: Server,
    title: "Metadata & Contextual Layer",
    description: "Processes accompanying data streams, including framerate, compression artifacts, and temporal correlations for comprehensive verification.",
  },
  {
    icon: Cpu,
    title: "High-Performance AI Model",
    description: "Utilizes a cutting-edge transformer model trained on diverse and large-scale multimodal datasets for industry-leading accuracy.",
  },
];

const technologies = [
  { title: "Vision Analysis", detail: "Convolutional Neural Networks (CNNs) for artifact identification." },
  { title: "Audio Forensics", detail: "Recurrent Neural Networks (RNNs) for voice pattern deviation." },
  { title: "Contextual Fusion", detail: "Gated Recurrent Units (GRUs) for blending data streams." },
];

// --- Components ---
const PrimaryButton = ({ children, className = '', onClick }) => (
  <button
    className={`
      px-8 py-3 text-lg font-semibold tracking-wide
      bg-indigo-600 hover:bg-indigo-700
      text-white rounded-xl shadow-lg
      transform transition duration-300 ease-in-out
      hover:scale-[1.02] active:scale-[0.98]
      focus:outline-none focus:ring-4 focus:ring-indigo-500 focus:ring-opacity-50
      ${className}
    `}
    onClick={onClick}
  >
    {children}
  </button>
);

const FeatureCard = ({ icon: Icon, title, description }) => (
  <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl transition duration-300 hover:bg-gray-700/50 border border-gray-700/50">
    <Icon className="w-10 h-10 text-indigo-400 mb-4" />
    <h3 className="text-xl font-bold text-white mb-3">{title}</h3>
    <p className="text-gray-400 leading-relaxed">{description}</p>
  </div>
);

const TechHighlight = ({ title, detail }) => (
  <div className="flex items-start space-x-4">
    <CheckCircle className="flex-shrink-0 w-6 h-6 text-emerald-400 mt-1" />
    <div>
      <h4 className="text-lg font-semibold text-white">{title}</h4>
      <p className="text-gray-400">{detail}</p>
    </div>
  </div>
);

const Header = ({ onLaunchClick }) => (
  <header className="fixed top-0 left-0 right-0 z-50 bg-gray-900/90 backdrop-blur-sm border-b border-gray-800">
    <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
      <div className="flex items-center space-x-2">
        <Zap className="w-6 h-6 text-indigo-500" />
        <span className="text-2xl font-extrabold text-white tracking-tight">
          DeepVerify
        </span>
      </div>
      <div className="hidden md:flex items-center space-x-6">
        {['Features', 'Technology', 'Pricing', 'Contact'].map((item) => (
          <a
            key={item}
            href={`#${item.toLowerCase()}`}
            className="text-gray-300 hover:text-indigo-400 transition duration-150 font-medium"
          >
            {item}
          </a>
        ))}
      </div>
      <PrimaryButton className="py-2 px-6" onClick={onLaunchClick}>
        Launch App
      </PrimaryButton>
    </nav>
  </header>
);

const Footer = () => (
  <footer className="bg-gray-950 border-t border-gray-800">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
      <div className="flex justify-center space-x-6 mb-6">
        {['Privacy Policy', 'Terms of Service', 'API Documentation'].map((item) => (
          <a key={item} href="#" className="text-gray-500 hover:text-indigo-400 text-sm">
            {item}
          </a>
        ))}
      </div>
      <p className="text-gray-600 text-sm">
        &copy; {new Date().getFullYear()} DeepVerify. All rights reserved. Built with the power of multimodal AI.
      </p>
    </div>
  </footer>
);

// Landing Page
const LandingPageContent = () => {
  const navigate = useNavigate();
  const handleLaunchClick = () => navigate('/upload'); // <-- navigate to Upload

  return (
    <div className="min-h-screen bg-gray-900 font-inter antialiased">
      <Header onLaunchClick={handleLaunchClick} />
      <main className="pt-20">
        {/* Hero Section */}
        <section className="relative pt-24 pb-32 overflow-hidden">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="inline-flex items-center text-sm font-medium text-indigo-300 bg-indigo-900/50 px-4 py-1 rounded-full mb-6 border border-indigo-700">
              <Lightbulb className="w-4 h-4 mr-2" />
              The Future of Trust
            </div>
            <h1 className="text-5xl md:text-7xl font-extrabold text-white mb-6 leading-tight md:leading-snug">
              Unmasking Deepfakes with <span className="text-indigo-400">Multimodal AI</span>
            </h1>
            <p className="max-w-3xl mx-auto text-xl text-gray-400 mb-10">
              Our revolutionary system fuses analysis from video, audio, and metadata channels to achieve unparalleled accuracy in deepfake detection, safeguarding against synthetic deception.
            </p>
            <div className="flex justify-center space-x-4">
              <PrimaryButton onClick={handleLaunchClick}>
                Start Free Detection <ArrowRight className="w-5 h-5 ml-2 inline-block" />
              </PrimaryButton>
              <a 
                href="#technology" 
                className="px-8 py-3 text-lg font-semibold rounded-xl text-indigo-400 border border-indigo-500 hover:bg-indigo-900/30 transition duration-300 shadow-lg"
              >
                See How It Works
              </a>
            </div>
          </div>
        </section>
        {/* Features Section */}
        <section id="features" className="py-20 md:py-32 bg-gray-950 border-t border-b border-gray-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-extrabold text-white mb-4">
                The Power of Three: Vision, Sound, Context
              </h2>
              <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                Deepfake models are constantly evolving. Our multimodal approach ensures we stay ahead of single-vector detection weaknesses.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {features.map((feature, index) => (
                <FeatureCard key={index} {...feature} />
              ))}
            </div>
          </div>
        </section>
        {/* Technology Section */}
        <section id="technology" className="py-20 md:py-32">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:flex lg:items-center lg:space-x-12">
              <div className="lg:w-1/2 mb-12 lg:mb-0">
                <h2 className="text-base font-semibold uppercase tracking-wider text-indigo-400 mb-3">
                  Our Methodology
                </h2>
                <h3 className="text-4xl font-extrabold text-white mb-6">
                  True Deepfake Detection is Comprehensive
                </h3>
                <p className="text-xl text-gray-400 mb-10">
                  Single-sensor analysis is easily fooled. DeepVerify is built on a sophisticated fusion engine that simultaneously processes and cross-validates data from multiple sources to eliminate false positives and catch subtle manipulations.
                </p>
                <div className="space-y-6">
                  {technologies.map((tech, index) => (
                    <TechHighlight key={index} {...tech} />
                  ))}
                </div>
              </div>
              <div className="lg:w-1/2 flex justify-center items-center">
                <div className="w-full h-80 bg-gray-800 rounded-3xl p-6 flex flex-col justify-between shadow-inner border border-indigo-700/50">
                  <div className="flex justify-between">
                    <div className="text-sm font-mono text-indigo-400">Multimodal Fusion Architecture</div>
                    <Shield className="w-6 h-6 text-emerald-500" />
                  </div>
                  <div className="flex justify-around items-center h-full">
                    <div className="text-center">
                      <Eye className="w-12 h-12 text-indigo-500 mx-auto mb-2" />
                      <p className="text-gray-300">Video Stream</p>
                    </div>
                    <ArrowRight className="w-8 h-8 text-gray-600" />
                    <div className="text-center">
                      <Mic className="w-12 h-12 text-indigo-500 mx-auto mb-2" />
                      <p className="text-gray-300">Audio Stream</p>
                    </div>
                    <ArrowRight className="w-8 h-8 text-gray-600" />
                    <div className="text-center">
                      <Server className="w-12 h-12 text-indigo-500 mx-auto mb-2" />
                      <p className="text-gray-300">Result</p>
                    </div>
                  </div>
                  <div className="text-xs text-gray-600 font-mono text-right">
                    Detection Latency: {'<'} 150ms
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        {/* Final CTA */}
        <section className="py-20 bg-indigo-900/20 border-t border-b border-indigo-900">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 className="text-4xl font-extrabold text-white mb-4">
              Ready to Trust Your Content Again?
            </h2>
            <p className="text-xl text-indigo-200 mb-8">
              Protect your brand, users, and digital reputation with the most advanced deepfake detection service available.
            </p>
            <PrimaryButton className="shadow-indigo-500/50" onClick={handleLaunchClick}>
              Get Started for Free Today
            </PrimaryButton>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

// ✅ App Component with Routes
const App = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<LandingPageContent />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/result" element={<Result />} />
    </Routes>
  </BrowserRouter>
);

export default App;