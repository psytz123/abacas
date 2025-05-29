'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Key, Plus, Trash2, Copy, Eye, EyeOff, Settings, Shield, Bell } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { toast } from 'react-hot-toast';

interface ApiKey {
  id: string;
  key: string;
  name: string;
  active: boolean;
  createdAt: string;
}

export function SettingsTab() {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set());
  const [notifications, setNotifications] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('30');

  useEffect(() => {
    fetchApiKeys();
  }, []);

  const fetchApiKeys = async () => {
    try {
      const response = await fetch('/api/api-keys');
      if (response.ok) {
        const keys = await response.json();
        setApiKeys(keys);
      }
    } catch (error) {
      console.error('Error fetching API keys:', error);
      toast.error('Failed to fetch API keys');
    }
  };

  const generateApiKey = async () => {
    if (!newKeyName.trim()) {
      toast.error('Please enter a name for the API key');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch('/api/api-keys', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: newKeyName.trim() }),
      });

      if (response.ok) {
        const newKey = await response.json();
        setApiKeys([newKey, ...apiKeys]);
        setNewKeyName('');
        toast.success('API key generated successfully');
        
        // Show the new key by default
        setVisibleKeys(new Set([newKey.id]));
      } else {
        const error = await response.json();
        toast.error(error.error || 'Failed to generate API key');
      }
    } catch (error) {
      console.error('Error generating API key:', error);
      toast.error('Failed to generate API key');
    } finally {
      setIsGenerating(false);
    }
  };

  const deleteApiKey = async (id: string) => {
    try {
      const response = await fetch(`/api/api-keys?id=${id}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setApiKeys(apiKeys.filter(key => key.id !== id));
        toast.success('API key deleted successfully');
      } else {
        toast.error('Failed to delete API key');
      }
    } catch (error) {
      console.error('Error deleting API key:', error);
      toast.error('Failed to delete API key');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('API key copied to clipboard');
  };

  const toggleKeyVisibility = (keyId: string) => {
    const newVisibleKeys = new Set(visibleKeys);
    if (newVisibleKeys.has(keyId)) {
      newVisibleKeys.delete(keyId);
    } else {
      newVisibleKeys.add(keyId);
    }
    setVisibleKeys(newVisibleKeys);
  };

  const maskApiKey = (key: string) => {
    return key.substring(0, 8) + '•'.repeat(key.length - 16) + key.substring(key.length - 8);
  };

  return (
    <div className="space-y-6">
      {/* API Keys Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm border-gray-200/50 dark:border-gray-700/50 shadow-lg">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Key className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <CardTitle>API Keys</CardTitle>
            </div>
            <CardDescription>
              Generate and manage API keys for external integrations
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Generate New API Key */}
            <div className="flex gap-2">
              <div className="flex-1">
                <Label htmlFor="keyName">API Key Name</Label>
                <Input
                  id="keyName"
                  placeholder="e.g., Mobile App, Dashboard Integration"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && generateApiKey()}
                />
              </div>
              <div className="flex items-end">
                <Button
                  onClick={generateApiKey}
                  disabled={isGenerating}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  {isGenerating ? 'Generating...' : 'Generate'}
                </Button>
              </div>
            </div>

            {/* API Keys List */}
            <div className="space-y-3">
              {apiKeys.length === 0 ? (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  <Key className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No API keys generated yet</p>
                  <p className="text-sm">Create your first API key to get started</p>
                </div>
              ) : (
                apiKeys.map((apiKey, index) => (
                  <motion.div
                    key={apiKey.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h4 className="font-medium text-gray-900 dark:text-white">
                            {apiKey.name}
                          </h4>
                          <Badge variant={apiKey.active ? "default" : "secondary"}>
                            {apiKey.active ? 'Active' : 'Inactive'}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="text-sm bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded font-mono">
                            {visibleKeys.has(apiKey.id) ? apiKey.key : maskApiKey(apiKey.key)}
                          </code>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleKeyVisibility(apiKey.id)}
                          >
                            {visibleKeys.has(apiKey.id) ? (
                              <EyeOff className="h-4 w-4" />
                            ) : (
                              <Eye className="h-4 w-4" />
                            )}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(apiKey.key)}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        </div>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          Created {new Date(apiKey.createdAt).toLocaleDateString()}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteApiKey(apiKey.id)}
                        className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* General Settings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm border-gray-200/50 dark:border-gray-700/50 shadow-lg">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5 text-green-600 dark:text-green-400" />
              <CardTitle>General Settings</CardTitle>
            </div>
            <CardDescription>
              Configure application preferences and behavior
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-base">Notifications</Label>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Receive alerts for mining events and status changes
                </p>
              </div>
              <Switch
                checked={notifications}
                onCheckedChange={setNotifications}
              />
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-base">Auto Refresh</Label>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Automatically refresh mining data
                </p>
              </div>
              <Switch
                checked={autoRefresh}
                onCheckedChange={setAutoRefresh}
              />
            </div>

            {autoRefresh && (
              <div className="space-y-2">
                <Label htmlFor="refreshInterval">Refresh Interval (seconds)</Label>
                <Input
                  id="refreshInterval"
                  type="number"
                  min="10"
                  max="300"
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(e.target.value)}
                  className="w-32"
                />
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Security Settings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm border-gray-200/50 dark:border-gray-700/50 shadow-lg">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-orange-600 dark:text-orange-400" />
              <CardTitle>Security</CardTitle>
            </div>
            <CardDescription>
              Manage security settings and access controls
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                <h4 className="font-medium text-orange-900 dark:text-orange-100">
                  API Key Security
                </h4>
              </div>
              <p className="text-sm text-orange-800 dark:text-orange-200">
                Keep your API keys secure and never share them publicly. 
                Regenerate keys if you suspect they have been compromised.
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
