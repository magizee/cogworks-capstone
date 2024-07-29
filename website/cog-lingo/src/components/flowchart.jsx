import React, { useState, useCallback } from 'react';
import ReactFlow, { 
  MiniMap, 
  Controls, 
  Background, 
  useNodesState, 
  useEdgesState,
  addEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Mic, AudioLines, AlignLeft, Podcast, BarChart3, MessageSquare } from 'lucide-react';

const nodeWidth = 180;
const nodeHeight = 80;

const steps = [
  { 
    id: '1', 
    type: 'customNode',
    position: { x: 0, y: 0 }, 
    data: { 
      label: 'Data Preparation', 
      details: 'Collect diverse datasets and fine-tune ASR system on accent-specific data.',
      icon: <AudioLines className="h-6 w-6 mr-2" />
    }
  },
  { 
    id: '2', 
    type: 'customNode',
    position: { x: 250, y: 0 }, 
    data: { 
      label: 'User Interaction', 
      details: 'User speaks and audio input is received for processing.',
      icon: <Mic className="h-6 w-6 mr-2" />
    }
  },
  { 
    id: '3', 
    type: 'customNode',
    position: { x: 125, y: 100 }, 
    data: { 
      label: 'ASR Implementation', 
      details: 'Pre-process audio, segment speech, transcribe using fine-tuned ASR, and post-process.',
      icon: <AlignLeft className="h-6 w-6 mr-2" />
    }
  },
  { 
    id: '4', 
    type: 'customNode',
    position: { x: 125, y: 200 }, 
    data: { 
      label: 'Phoneme Analysis', 
      details: 'Convert speech to phonemes, align with standard phonemes using DTW, compare and identify mismatches.',
      icon: <Podcast className="h-6 w-6 p-10 mr-2" />
    }
  },
  { 
    id: '5', 
    type: 'customNode',
    position: { x: 125, y: 300 }, 
    data: { 
      label: 'Scoring & Feedback', 
      details: 'Calculate pronunciation score, generate visual feedback and prepare improvement tips.',
      icon: <BarChart3 className="h-6 w-6 mr-2" />
    }
  },
  { 
    id: '6', 
    type: 'customNode',
    position: { x: 125, y: 400 }, 
    data: { 
      label: 'User Experience', 
      details: 'Display score and feedback to user. User reviews results and can try again.',
      icon: <MessageSquare className="h-6 w-6 mr-2" />
    }
  },
];

const initialEdges = [
  { id: 'e1-3', source: '1', target: '3' },
  { id: 'e2-3', source: '2', target: '3' },
  { id: 'e3-4', source: '3', target: '4' },
  { id: 'e4-5', source: '4', target: '5' },
  { id: 'e5-6', source: '5', target: '6' },
  { id: 'e6-2', source: '6', target: '2', type: 'step', style: { stroke: '#FF0000' } },
];

const CustomNode = ({ data, onClick }) => {
  return (
    <div 
      className="custom-node"
      style={{
        width: nodeWidth,
        height: nodeHeight,
        transition: 'all 0.3s ease',
        cursor: 'pointer'
      }}
      onClick={onClick}
    >
      <Card className="w-full h-full transition-all duration-300">
        <CardHeader className="p-3">
          <div className="flex items-center">
            {data.icon}
            <CardTitle className="text-sm">{data.label}</CardTitle>
          </div>
        </CardHeader>
      </Card>
    </div>
  );
};

const nodeTypes = {
  customNode: CustomNode,
};

const InteractivePronunciationAssessmentFlowchart = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(steps);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [activeStep, setActiveStep] = useState(null);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const handleNodeClick = (node) => {
    setActiveStep(node.data);
  };

  return (
    <div style={{ display: 'flex', width: '100%', height: '600px' }}>
      <div style={{ width: '75%', height: '100%' }}>
        <ReactFlow
          nodes={nodes.map(node => ({ ...node, data: { ...node.data, onClick: () => handleNodeClick(node) } }))}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>
      <div style={{ width: '25%', padding: '10px' }}>
        {activeStep ? (
          <Card className="transition-all duration-300">
            <CardHeader>
              <div className="flex items-center">
                {activeStep.icon}
                <CardTitle className="text-lg ml-2">{activeStep.label}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p>{activeStep.details}</p>
            </CardContent>
          </Card>
        ) : (
          <p>Select a node to see details</p>
        )}
      </div>
    </div>
  );
};

export default InteractivePronunciationAssessmentFlowchart;
