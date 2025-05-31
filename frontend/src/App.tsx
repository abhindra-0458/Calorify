import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Github, Upload, Utensils } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [calories, setCalories] = useState<number | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
        setCalories(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const processImage = async () => {
    if (!image) return;
  
    setIsProcessing(true);
    setCalories(null);
  
    try {
      const formData = new FormData();
      const blob = await fetch(image).then(res => res.blob());
      formData.append("image", blob, "uploaded-image.jpg");
  
      const response = await fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        body: formData
      });
  
      if (!response.ok) throw new Error("Failed to process image");
  
      const data = await response.json();
      setImage(data.image_url);
      setCalories(data.calories);
  
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setIsProcessing(false);
    }
  };
  

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-green-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm" onClick={() => window.location.reload()}>
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Utensils className="h-8 w-8 text-green-600" />
            <h1 className="text-3xl font-bold text-gray-900">Calorify</h1>
          </div>
        </div>
      </header>


      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
            <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-green-400'}`}>
              <input {...getInputProps()} />
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-lg text-gray-600">
                {isDragActive ? 'Drop your food image here' : 'Drag & drop a food image, or click to select'}
              </p>
              <p className="text-sm text-gray-500 mt-2">Supported formats: JPEG, PNG</p>
            </div>

            {image && (
              <div className="mt-8 space-y-6">
                <div className="relative pb-[56.25%]">
                  <img 
                    src={image} 
                    alt="Food" 
                    className="absolute inset-0 w-full h-full object-contain rounded-lg"
                  />
                </div>
                <div className="flex justify-center">
                  <button
                    onClick={processImage}
                    disabled={isProcessing}
                    className={`px-6 py-3 rounded-full text-white font-semibold transition-colors ${isProcessing ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'}`}
                  >
                    {isProcessing ? 'Processing...' : 'Analyze Image'}
                  </button>
                </div>
              </div>
            )}

            {calories !== null && (
              <div className="mt-8 p-6 bg-green-50 rounded-lg">
                <h3 className="text-xl font-semibold text-green-800 mb-2">Analysis Results</h3>
                <p className="text-3xl font-bold text-green-600">{Math.floor(calories)} calories</p>
                <p className="text-sm text-green-700 mt-2">Estimated calorie content based on image analysis</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-center items-center space-x-4">
            <a
              href="https://github.com/yourusername/calorify"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <Github className="h-5 w-5" />
              <span>View on GitHub</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;