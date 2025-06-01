import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Upload, BarChart3 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface FormData {
  question: string;
  city: string;
  purpose: string;
  rooms: number;
  type: string;
  budget: string;
  topN: number;
  investmentTimeline: number;
  method: string;
  additionalInfo: string;
  skipDefineObjectiveStep: boolean;
  useHumanInTheLoop: boolean;
  file?: File;
}

const PropertyAnalysisForm = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<FormData>({
    question: 'Can you find underrated 2-bedroom apartment units?',
    city: 'Melbourne',
    purpose: 'investment',
    rooms: 2,
    type: 'house',
    budget: '600k-900k',
    topN: 10,
    investmentTimeline: 10,
    method: 'Create a scoring function using all of the following features: price_per_sqm, building_area, how_many_cars, how_many_bathrooms, year_built, distance, type',
    additionalInfo: 'I\'ll leave it to you to figure out what are underrated items and how to create the scoring function.',
    skipDefineObjectiveStep: false,
    useHumanInTheLoop: true,
  });
  const [file, setFile] = useState<File | null>(null);

  const handleInputChange = (field: keyof FormData, value: string | number | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Store form data in localStorage for access across pages
    localStorage.setItem('analysisFormData', JSON.stringify({ ...formData, fileName: file?.name }));
    // Navigate to step 1
    navigate('/analysis/step1');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <BarChart3 className="h-12 w-12 text-blue-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900">Property Analysis AI</h1>
          </div>
          <p className="text-xl text-gray-600">Discover investment opportunities with AI-powered analysis</p>
        </div>

        <Card className="shadow-xl">
          <CardHeader>
            <CardTitle className="text-2xl text-center">Investment Analysis Request</CardTitle>
            <CardDescription className="text-center">
              Fill out the form below to start your property analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="question">Main Question</Label>
                  <Textarea
                    id="question"
                    placeholder="e.g., Can you find underrated 2-bedroom apartment units?"
                    value={formData.question}
                    onChange={(e) => handleInputChange('question', e.target.value)}
                    className="min-h-[100px]"
                    required
                  />
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="city">City</Label>
                    <Input
                      id="city"
                      placeholder="e.g., Melbourne"
                      value={formData.city}
                      onChange={(e) => handleInputChange('city', e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="purpose">Purpose</Label>
                    <Select value={formData.purpose} onValueChange={(value) => handleInputChange('purpose', value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select purpose" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="investment">Investment</SelectItem>
                        <SelectItem value="personal">Personal Use</SelectItem>
                        <SelectItem value="rental">Rental</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="rooms">Number of Rooms</Label>
                  <Input
                    id="rooms"
                    type="number"
                    min="1"
                    value={formData.rooms}
                    onChange={(e) => handleInputChange('rooms', parseInt(e.target.value))}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="type">Property Type</Label>
                  <Select value={formData.type} onValueChange={(value) => handleInputChange('type', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="house">House</SelectItem>
                      <SelectItem value="apartment">Apartment</SelectItem>
                      <SelectItem value="unit">Unit</SelectItem>
                      <SelectItem value="townhouse">Townhouse</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="budget">Budget Range</Label>
                  <Input
                    id="budget"
                    placeholder="e.g., 600k-900k"
                    value={formData.budget}
                    onChange={(e) => handleInputChange('budget', e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="topN">Top N Results</Label>
                  <Input
                    id="topN"
                    type="number"
                    min="1"
                    max="100"
                    value={formData.topN}
                    onChange={(e) => handleInputChange('topN', parseInt(e.target.value))}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="investmentTimeline">Investment Timeline (Years)</Label>
                  <Input
                    id="investmentTimeline"
                    type="number"
                    min="1"
                    value={formData.investmentTimeline}
                    onChange={(e) => handleInputChange('investmentTimeline', parseInt(e.target.value))}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="method">Analysis Method</Label>
                <Textarea
                  id="method"
                  placeholder="e.g., Create a scoring function using price_per_sqm, building_area, distance..."
                  value={formData.method}
                  onChange={(e) => handleInputChange('method', e.target.value)}
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="additionalInfo">Additional Information</Label>
                <Textarea
                  id="additionalInfo"
                  placeholder="Any additional requirements or preferences..."
                  value={formData.additionalInfo}
                  onChange={(e) => handleInputChange('additionalInfo', e.target.value)}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="p-4">
                  <div className="flex items-center justify-between space-x-2">
                    <div className="space-y-1">
                      <Label htmlFor="skipDefineObjectiveStep">Skip Define Objective Step</Label>
                      <p className="text-sm text-gray-600">If this is turned on, the agent won't ask you clarification question about your request.</p>
                    </div>
                    <Switch
                      id="skipDefineObjectiveStep"
                      checked={formData.skipDefineObjectiveStep}
                      onCheckedChange={(checked) => handleInputChange('skipDefineObjectiveStep', checked)}
                    />
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="flex items-center justify-between space-x-2">
                    <div className="space-y-1">
                      <Label htmlFor="useHumanInTheLoop">Use Human in the Loop</Label>
                      <p className="text-sm text-gray-600">Every time the agent is not sure about something during the analysis, it will ask you for clarification or review the result.</p>
                    </div>
                    <Switch
                      id="useHumanInTheLoop"
                      checked={formData.useHumanInTheLoop}
                      onCheckedChange={(checked) => handleInputChange('useHumanInTheLoop', checked)}
                    />
                  </div>
                </Card>
              </div>

              <div className="space-y-2">
                <Label htmlFor="file">Upload Data File (CSV)</Label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                  <input
                    type="file"
                    id="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <label htmlFor="file" className="cursor-pointer">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">
                      {file ? file.name : 'Click to upload CSV file or drag and drop'}
                    </p>
                  </label>
                </div>
              </div>

              <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-lg py-6">
                Start Analysis
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PropertyAnalysisForm;
