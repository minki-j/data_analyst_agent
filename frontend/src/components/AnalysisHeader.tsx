
import React from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { BarChart3, ChevronLeft, ChevronRight } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';

interface AnalysisHeaderProps {
  currentStep: number;
  totalSteps: number;
}

const AnalysisHeader = ({ currentStep, totalSteps }: AnalysisHeaderProps) => {
  const navigate = useNavigate();
  const location = useLocation();

  const steps = [
    { name: 'Define Objective', path: '/analysis/step1' },
    { name: 'Data Cleaning', path: '/analysis/step2' },
    { name: 'Data Exploration', path: '/analysis/step3' },
    { name: 'Data Analysis', path: '/analysis/step4' },
    { name: 'Report', path: '/analysis/step5' },
  ];

  const progress = (currentStep / totalSteps) * 100;

  const goToPreviousStep = () => {
    if (currentStep > 1) {
      navigate(steps[currentStep - 2].path);
    } else {
      navigate('/');
    }
  };

  const goToNextStep = () => {
    if (currentStep < totalSteps) {
      navigate(steps[currentStep].path);
    }
  };

  return (
    <div className="bg-white shadow-lg border-b">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <BarChart3 className="h-8 w-8 text-blue-600 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900">Property Analysis</h1>
          </div>

          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={goToPreviousStep}
              className="flex items-center"
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              {currentStep === 1 ? 'Back to Form' : 'Previous'}
            </Button>

            {currentStep < totalSteps && (
              <Button
                variant="outline"
                size="sm"
                onClick={goToNextStep}
                className="flex items-center"
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between mb-2">
          <div className="flex space-x-4">
            {steps.map((step, index) => (
              <button
                key={index}
                onClick={() => navigate(step.path)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${index + 1 === currentStep
                  ? 'bg-blue-100 text-blue-700'
                  : index + 1 < currentStep
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-500'
                  }`}
              >
                {index + 1}. {step.name}
              </button>
            ))}
          </div>

        </div>

        {/* <Progress value={progress} className="h-2" /> */}
      </div>
    </div>
  );
};

export default AnalysisHeader;
